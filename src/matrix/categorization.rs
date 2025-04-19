//! # Row Categorization for the MAGNUS Algorithm
//!
//! This module implements the logic for categorizing rows based on
//! computational characteristics as described in Section 3.1 of the MAGNUS paper.
//!
//! ## Categorization Logic
//!
//! The MAGNUS algorithm categorizes each row of matrix A into one of four categories
//! based on its computational characteristics:
//!
//! 1. **Sort**: When the intermediate product has a small number of non-zeros.
//!    - The intermediate nnz is less than or equal to `dense_accum_threshold`
//!    - Uses a sort-based accumulation approach
//!
//! 2. **DenseAccumulation**: When the dense accumulation array fits in L2 cache.
//!    - The dense array size (num_cols * sizeof(value)) is less than or equal to L2 cache size
//!    - Uses a dense array-based accumulation
//!
//! 3. **FineLevel**: When fine-level reordering structures fit in L2 cache.
//!    - Histogram + prefix sum + reordered cols/vals fit in L2 cache
//!    - Uses chunked column-major ordering for improved locality
//!
//! 4. **CoarseLevel**: When structures don't fit in L2 cache.
//!    - Combines coarse-level batching with fine-level reordering
//!    - Can be disabled via config (falling back to FineLevel)
//!
//! The categorization process takes into account:
//! - Matrix structure and density
//! - Hardware parameters (cache size)
//! - User configuration

use num_traits::Num;
use std::ops::AddAssign;

use crate::matrix::config::{MagnusConfig, RowCategory};
use crate::matrix::SparseMatrixCSR;

/// Analyzes the rows of matrices A and B and categorizes them for efficient processing
///
/// This implementation follows the categorization logic in Section 3.1 of the paper:
/// 1. Small intermediate products: Use sort-based method
/// 2. Intermediate product fits in L2 cache: Use dense accumulation
/// 3. Fine-level reordering
/// 4. Coarse-level reordering
pub fn categorize_rows<T>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    config: &MagnusConfig,
) -> Vec<RowCategory>
where
    T: Copy + Num + AddAssign,
{
    assert_eq!(
        a.n_cols, b.n_rows,
        "Matrix dimensions must be compatible for multiplication"
    );

    let mut categories = Vec::with_capacity(a.n_rows);

    // Get the critical parameters from the config
    let dense_threshold = config.dense_accum_threshold;
    let l2_cache_size = config.system_params.l2_cache_size;
    let _cache_line_bytes = config.system_params.cache_line_size;

    // Estimate size of data types
    let sizeof_idx = std::mem::size_of::<usize>();
    let sizeof_value = std::mem::size_of::<T>();

    // Fixed memory overhead for fine-level structures (using 2 as default chunk_log)
    let chunk_log = 2; // Default from paper, will be tuned later
    let chunk_size = 1 << chunk_log;
    let n_chunks = (b.n_cols + chunk_size - 1) / chunk_size;

    // Iterate through each row of A and categorize it
    for i in 0..a.n_rows {
        // Get row i of A
        let row_start = a.row_ptr[i];
        let row_end = a.row_ptr[i + 1];
        let nnz_row_a = row_end - row_start;

        if nnz_row_a == 0 {
            // Empty row, no computation needed
            categories.push(RowCategory::Sort);
            continue;
        }

        // Estimate the intermediate product size (number of entries)
        // by summing the number of nonzeros in each corresponding row of B
        let mut intermediate_nnz = 0;
        for a_idx in row_start..row_end {
            let k = a.col_idx[a_idx];
            intermediate_nnz += b.row_ptr[k + 1] - b.row_ptr[k];
        }

        // 1. If intermediate product is small, use sort-based method
        if intermediate_nnz <= dense_threshold {
            categories.push(RowCategory::Sort);
            continue;
        }

        // 2. If intermediate product fits in L2 cache, use dense accumulation
        // Dense array size in bytes: number of columns * sizeof(value)
        let dense_size = b.n_cols * sizeof_value;
        if dense_size <= l2_cache_size {
            categories.push(RowCategory::DenseAccumulation);
            continue;
        }

        // 3. Check if fine-level reordering structures fit in L2 cache
        // Fine level structures: counts, offsets (each n_chunks),
        //                        reordered cols/vals (intermediate_nnz each)
        let fine_level_size =
            2 * n_chunks * sizeof_idx + 2 * intermediate_nnz * (sizeof_idx + sizeof_value);

        if fine_level_size <= l2_cache_size {
            categories.push(RowCategory::FineLevel);
            continue;
        }

        // 4. If we get here, use coarse-level reordering
        if config.enable_coarse_level {
            categories.push(RowCategory::CoarseLevel);
        } else {
            // If coarse level is disabled, fall back to fine level
            categories.push(RowCategory::FineLevel);
        }
    }

    categories
}

/// Analyzes matrices A and B and returns a summary of row categorization
///
/// This is useful for understanding the computational characteristics
/// of a specific matrix multiplication problem.
pub fn analyze_categorization<T>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    config: &MagnusConfig,
) -> CategorizationSummary
where
    T: Copy + Num + AddAssign,
{
    let categories = categorize_rows(a, b, config);

    let mut summary = CategorizationSummary::default();
    summary.total_rows = categories.len();

    for category in categories {
        match category {
            RowCategory::Sort => summary.sort_count += 1,
            RowCategory::DenseAccumulation => summary.dense_count += 1,
            RowCategory::FineLevel => summary.fine_level_count += 1,
            RowCategory::CoarseLevel => summary.coarse_level_count += 1,
        }
    }

    summary
}

/// Summary of row categorization for sparse matrix multiplication
#[derive(Debug, Default, Clone, Copy)]
pub struct CategorizationSummary {
    /// Total number of rows analyzed
    pub total_rows: usize,
    /// Number of rows using sort-based accumulation
    pub sort_count: usize,
    /// Number of rows using dense accumulation
    pub dense_count: usize,
    /// Number of rows using fine-level reordering
    pub fine_level_count: usize,
    /// Number of rows using coarse-level reordering
    pub coarse_level_count: usize,
}

impl CategorizationSummary {
    /// Returns the percentage of rows in each category
    pub fn percentages(&self) -> (f64, f64, f64, f64) {
        let total = self.total_rows as f64;
        if total == 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        (
            (self.sort_count as f64) / total * 100.0,
            (self.dense_count as f64) / total * 100.0,
            (self.fine_level_count as f64) / total * 100.0,
            (self.coarse_level_count as f64) / total * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorize_small_matrices() {
        // Create small test matrices that will all use sort-based accumulation
        let a = SparseMatrixCSR::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1, 2, 3]);

        let b = SparseMatrixCSR::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![4, 5, 6]);

        let config = MagnusConfig::default();

        // Categorize rows
        let categories = categorize_rows(&a, &b, &config);

        // All rows should use sort-based accumulation (small intermediate products)
        assert_eq!(categories.len(), 3);
        assert_eq!(categories[0], RowCategory::Sort);
        assert_eq!(categories[1], RowCategory::Sort);
        assert_eq!(categories[2], RowCategory::Sort);
    }

    #[test]
    fn test_empty_row_categorization() {
        // Create a matrix with an empty row
        let a = SparseMatrixCSR::new(
            3,
            3,
            vec![0, 1, 1, 2], // Row 1 is empty
            vec![0, 2],
            vec![1, 3],
        );

        let b = SparseMatrixCSR::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![4, 5, 6]);

        let config = MagnusConfig::default();

        // Categorize rows
        let categories = categorize_rows(&a, &b, &config);

        // Empty row should still be categorized (as Sort is most efficient)
        assert_eq!(categories.len(), 3);
        assert_eq!(categories[1], RowCategory::Sort);
    }

    #[test]
    fn test_categorization_respects_threshold() {
        // Create matrices where one row produces exactly the threshold number
        // of intermediate products

        // Set up config with threshold of 4
        let mut config = MagnusConfig::default();
        config.dense_accum_threshold = 4;

        // A has one row with 2 non-zeros
        let a = SparseMatrixCSR::new(2, 5, vec![0, 1, 3], vec![0, 1, 3], vec![1, 2, 3]);

        // B has rows with specific numbers of non-zeros to test the threshold
        // Row 0: 1 non-zero
        // Row 1: 3 non-zeros
        // Row 3: 1 non-zero
        // Total for row 1 of A: 1 + 3 = 4 (exactly threshold)
        let b = SparseMatrixCSR::new(
            5,
            5,
            vec![0, 1, 4, 4, 5, 5],
            vec![0, 0, 2, 4, 0],
            vec![1, 2, 3, 4, 5],
        );

        // Categorize rows
        let categories = categorize_rows(&a, &b, &config);

        // Row 0 of A should use Sort (1 intermediate nnz)
        assert_eq!(categories[0], RowCategory::Sort);

        // Row 1 of A should use Sort (at threshold)
        assert_eq!(categories[1], RowCategory::Sort);

        // Now increase nnz in row 1 of B to push over threshold
        let b2 = SparseMatrixCSR::new(
            5,
            5,
            vec![0, 1, 5, 5, 6, 6], // Row 1 now has 4 non-zeros
            vec![0, 0, 1, 2, 4, 0],
            vec![1, 2, 3, 4, 5, 6],
        );

        // Categorize again
        let categories2 = categorize_rows(&a, &b2, &config);

        // Row 0 of A should still use Sort
        assert_eq!(categories2[0], RowCategory::Sort);

        // Row 1 of A should not use Sort (exceeds threshold)
        assert_ne!(categories2[1], RowCategory::Sort);
    }

    #[test]
    fn test_categorization_summary() {
        // Create small test matrices
        let a = SparseMatrixCSR::new(
            4,
            3,
            vec![0, 1, 2, 3, 4],
            vec![0, 1, 2, 0],
            vec![1, 2, 3, 4],
        );

        let b = SparseMatrixCSR::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![4, 5, 6]);

        let config = MagnusConfig::default();

        // Get categorization summary
        let summary = analyze_categorization(&a, &b, &config);

        // Check total count
        assert_eq!(summary.total_rows, 4);

        // All should be sort in this simple example
        assert_eq!(summary.sort_count, 4);

        // Check percentages
        let (sort_pct, dense_pct, fine_pct, coarse_pct) = summary.percentages();
        assert_eq!(sort_pct, 100.0);
        assert_eq!(dense_pct, 0.0);
        assert_eq!(fine_pct, 0.0);
        assert_eq!(coarse_pct, 0.0);
    }

    #[test]
    fn test_coarse_level_disabled() {
        // Create matrices that would need coarse-level reordering
        let a = SparseMatrixCSR::new(
            1,
            5000, // One row with many potential connections
            vec![0, 5000],
            (0..5000).collect(), // Connect to all rows in B
            vec![1; 5000],
        );

        let b = SparseMatrixCSR::new(
            5000,
            5000,
            (0..=5000).collect(), // Each row has 1 non-zero
            (0..5000).collect(),
            vec![1; 5000],
        );

        // Use a modified config with a smaller L2 cache to ensure coarse level is needed
        let mut config_enabled = MagnusConfig::default();
        config_enabled.system_params.l2_cache_size = 10000; // Very small cache

        let categories_enabled = categorize_rows(&a, &b, &config_enabled);

        // With small L2 cache and large intermediate products, row should use CoarseLevel
        assert_eq!(categories_enabled[0], RowCategory::CoarseLevel);

        // Now with coarse-level disabled but same small cache
        let mut config_disabled = config_enabled.clone();
        config_disabled.enable_coarse_level = false;

        let categories_disabled = categorize_rows(&a, &b, &config_disabled);

        // Row should use FineLevel instead
        assert_eq!(categories_disabled[0], RowCategory::FineLevel);
    }
}
