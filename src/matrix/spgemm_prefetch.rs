//! SpGEMM with memory prefetching optimization
//!
//! This module implements sparse matrix multiplication with strategic
//! memory prefetching to improve cache performance.

use crate::accumulator::Accumulator;
use crate::constants::*;
use crate::matrix::SparseMatrixCSR;
use crate::utils::prefetch::{
    prefetch_read_l1, prefetch_read_l2, PrefetchConfig, PrefetchStrategy,
};
use num_traits::Num;
use std::ops::AddAssign;

/// Multiply two sparse matrices with prefetching optimization
pub fn spgemm_with_prefetch<T>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    config: &PrefetchConfig,
) -> SparseMatrixCSR<T>
where
    T: Num + Copy + AddAssign + Send + Sync + 'static,
{
    assert_eq!(
        a.n_cols, b.n_rows,
        "Matrix dimensions don't match for multiplication"
    );

    let n_rows = a.n_rows;
    let n_cols = b.n_cols;

    // Result matrix in CSR format
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    // Process each row of A
    for i in 0..n_rows {
        // Apply prefetch strategy
        apply_prefetch_strategy(a, b, i, config);

        // Create accumulator for this row (use sort-based for now)
        let mut accumulator =
            crate::accumulator::sort::SortAccumulator::<T>::new(DEFAULT_SORT_ACCUMULATOR_SIZE);

        // Get row i of A
        let a_row_start = a.row_ptr[i];
        let a_row_end = a.row_ptr[i + 1];

        // For each non-zero in row i of A
        for a_idx in a_row_start..a_row_end {
            let k = a.col_idx[a_idx];
            let a_val = a.values[a_idx];

            // Prefetch upcoming B row if using moderate/aggressive strategy
            if config.enabled && config.strategy != PrefetchStrategy::Conservative {
                prefetch_b_row(b, k, config.distance);
            }

            // Get row k of B
            let b_row_start = b.row_ptr[k];
            let b_row_end = b.row_ptr[k + 1];

            // For each non-zero in row k of B
            for b_idx in b_row_start..b_row_end {
                let j = b.col_idx[b_idx];
                let b_val = b.values[b_idx];

                // Prefetch write location in accumulator if aggressive
                if config.enabled && config.strategy == PrefetchStrategy::Aggressive {
                    // Hint: we'll write to accumulator soon
                    // (In practice, accumulator handles this internally)
                }

                // Multiply and accumulate
                let product = a_val * b_val;
                accumulator.accumulate(j, product);
            }
        }

        // Extract results from accumulator
        let (row_cols, row_vals) = accumulator.extract_result();

        // Append to result
        for (col, val) in row_cols.into_iter().zip(row_vals.into_iter()) {
            col_idx.push(col);
            values.push(val);
        }

        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(n_rows, n_cols, row_ptr, col_idx, values)
}

/// Apply prefetch strategy for current row
#[inline(always)]
fn apply_prefetch_strategy<T>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    current_row: usize,
    config: &PrefetchConfig,
) where
    T: Copy,
{
    if !config.enabled {
        return;
    }

    match config.strategy {
        PrefetchStrategy::None => {}

        PrefetchStrategy::Conservative => {
            // Prefetch next row of A
            if current_row + 1 < a.n_rows {
                let next_row_start = a.row_ptr[current_row + 1];
                if next_row_start < a.col_idx.len() {
                    unsafe {
                        prefetch_read_l1(a.col_idx.as_ptr().add(next_row_start));
                        prefetch_read_l1(a.values.as_ptr().add(next_row_start));
                    }
                }
            }
        }

        PrefetchStrategy::Moderate => {
            // Prefetch next row of A
            if current_row + 1 < a.n_rows {
                let next_row_start = a.row_ptr[current_row + 1];
                if next_row_start < a.col_idx.len() {
                    unsafe {
                        prefetch_read_l1(a.col_idx.as_ptr().add(next_row_start));
                        prefetch_read_l1(a.values.as_ptr().add(next_row_start));
                    }
                }
            }

            // Prefetch first few columns from current row of A
            // to prepare B row lookups
            let row_start = a.row_ptr[current_row];
            let row_end = a.row_ptr[current_row + 1];
            let prefetch_count = (row_end - row_start).min(config.distance);

            for offset in 0..prefetch_count {
                if row_start + offset < a.col_idx.len() {
                    let col = a.col_idx[row_start + offset];
                    if col < b.n_rows {
                        // Prefetch row pointers of B
                        unsafe {
                            prefetch_read_l2(b.row_ptr.as_ptr().add(col));
                        }
                    }
                }
            }
        }

        PrefetchStrategy::Aggressive => {
            // Prefetch next several rows of A
            for dist in 1..=config.distance {
                if current_row + dist < a.n_rows {
                    let future_row_start = a.row_ptr[current_row + dist];
                    if future_row_start < a.col_idx.len() {
                        unsafe {
                            // Use L2 for farther ahead
                            if dist <= 2 {
                                prefetch_read_l1(a.col_idx.as_ptr().add(future_row_start));
                                prefetch_read_l1(a.values.as_ptr().add(future_row_start));
                            } else {
                                prefetch_read_l2(a.col_idx.as_ptr().add(future_row_start));
                                prefetch_read_l2(a.values.as_ptr().add(future_row_start));
                            }
                        }
                    }
                }
            }

            // Aggressively prefetch B rows
            let row_start = a.row_ptr[current_row];
            let row_end = a.row_ptr[current_row + 1];
            let prefetch_count = (row_end - row_start).min(config.distance * 2);

            for offset in 0..prefetch_count {
                if row_start + offset < a.col_idx.len() {
                    let col = a.col_idx[row_start + offset];
                    prefetch_b_row(b, col, 2);
                }
            }
        }

        PrefetchStrategy::Adaptive => {
            // In production, would use AccessPatternAnalyzer
            // For now, use moderate strategy
            apply_prefetch_strategy(a, b, current_row, &PrefetchConfig::moderate());
        }
    }
}

/// Prefetch a row from matrix B
#[inline(always)]
fn prefetch_b_row<T>(b: &SparseMatrixCSR<T>, row: usize, distance: usize)
where
    T: Copy,
{
    if row >= b.n_rows {
        return;
    }

    let row_start = b.row_ptr[row];
    let row_end = b.row_ptr[row + 1];

    // Prefetch the row data
    let prefetch_count = (row_end - row_start).min(distance);

    for offset in 0..prefetch_count {
        if row_start + offset < b.col_idx.len() {
            unsafe {
                prefetch_read_l1(b.col_idx.as_ptr().add(row_start + offset));
                prefetch_read_l1(b.values.as_ptr().add(row_start + offset));
            }
        }
    }
}

/// Analyze memory access pattern and recommend prefetch strategy
pub fn analyze_and_recommend<T>(a: &SparseMatrixCSR<T>, b: &SparseMatrixCSR<T>) -> PrefetchStrategy
where
    T: Copy,
{
    // Simple heuristics based on matrix characteristics

    let a_nnz = a.values.len();
    let b_nnz = b.values.len();

    let a_density = a_nnz as f64 / (a.n_rows * a.n_cols) as f64;
    let b_density = b_nnz as f64 / (b.n_rows * b.n_cols) as f64;

    // Average non-zeros per row
    let a_avg_nnz = a_nnz / a.n_rows.max(1);
    let b_avg_nnz = b_nnz / b.n_rows.max(1);

    // If matrices are very sparse, prefetching has less benefit
    if a_density < SPARSE_DENSITY_THRESHOLD && b_density < SPARSE_DENSITY_THRESHOLD {
        return PrefetchStrategy::Conservative;
    }

    // If B has many non-zeros per row, aggressive prefetch helps
    if b_avg_nnz >= B_MATRIX_AVG_NNZ_THRESHOLD {
        return PrefetchStrategy::Aggressive;
    }

    // If matrices are moderately sparse
    if a_avg_nnz > MIN_AVG_NNZ_THRESHOLD && b_avg_nnz > MIN_AVG_NNZ_THRESHOLD {
        return PrefetchStrategy::Moderate;
    }

    // Default to conservative
    PrefetchStrategy::Conservative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spgemm_with_prefetch() {
        // Create small test matrices
        let a = SparseMatrixCSR::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        let b = SparseMatrixCSR::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![1, 2, 0, 2, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        // Test with different prefetch strategies
        let configs = [
            PrefetchConfig::none(),
            PrefetchConfig::conservative(),
            PrefetchConfig::moderate(),
            PrefetchConfig::aggressive(),
        ];

        for config in &configs {
            let result = spgemm_with_prefetch(&a, &b, config);

            // Verify dimensions
            assert_eq!(result.n_rows, 3);
            assert_eq!(result.n_cols, 3);

            // Verify result has expected non-zeros
            assert!(result.values.len() > 0);
        }
    }

    #[test]
    fn test_prefetch_recommendation() {
        // Dense matrices should get aggressive prefetch
        // Create proper row_ptr where each row has 100 non-zeros (total 10000)
        let row_ptr: Vec<usize> = (0..=100).map(|i| i * 100).collect();
        let dense_a = SparseMatrixCSR::new(
            100,
            100,
            row_ptr,
            (0..10000).map(|i| i % 100).collect(),
            vec![1.0; 10000],
        );

        let strategy = analyze_and_recommend(&dense_a, &dense_a);
        assert_eq!(strategy, PrefetchStrategy::Aggressive);

        // Very sparse matrices should get conservative
        let sparse_a = SparseMatrixCSR::new(
            1000,
            1000,
            (0..=1000).collect(),
            (0..1000).collect(),
            vec![1.0; 1000],
        );

        let strategy = analyze_and_recommend(&sparse_a, &sparse_a);
        assert_eq!(strategy, PrefetchStrategy::Conservative);
    }
}
