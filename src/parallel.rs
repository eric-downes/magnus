//! # Parallel Implementation of MAGNUS SpGEMM
//!
//! This module provides parallel implementations of the MAGNUS algorithm
//! using Rayon for parallel processing of matrix rows.

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use crate::accumulator;
use crate::matrix;
use crate::reordering;

/// Type alias for batch results accumulator
type BatchResults<T> = Arc<Mutex<Vec<Vec<(usize, T)>>>>;

/// Performs sparse general matrix-matrix multiplication (SpGEMM)
/// using the MAGNUS algorithm with parallel row processing.
///
/// This function parallelizes the row processing phase of the MAGNUS algorithm,
/// allowing for better utilization of multi-core processors.
///
/// # Arguments
///
/// * `a` - Left input matrix in CSR format
/// * `b` - Right input matrix in CSR format
/// * `config` - Configuration parameters for the MAGNUS algorithm
///
/// # Returns
///
/// The result matrix C = A×B in CSR format
///
/// # Examples
///
/// ```
/// use magnus::{SparseMatrixCSR, MagnusConfig, magnus_spgemm_parallel};
///
/// // Create two sparse matrices
/// let a = SparseMatrixCSR::<f64>::new(
///     2, 2,
///     vec![0, 1, 2],
///     vec![0, 1],
///     vec![1.0, 1.0],
/// );
///
/// let b = SparseMatrixCSR::<f64>::new(
///     2, 2,
///     vec![0, 1, 2],
///     vec![0, 1],
///     vec![2.0, 2.0],
/// );
///
/// // Multiply the matrices using parallel MAGNUS
/// let config = MagnusConfig::default();
/// let c = magnus_spgemm_parallel(&a, &b, &config);
///
/// // Result should be a diagonal matrix with elements 2.0
/// assert_eq!(c.n_rows, 2);
/// assert_eq!(c.n_cols, 2);
/// ```
pub fn magnus_spgemm_parallel<T>(
    a: &matrix::SparseMatrixCSR<T>,
    b: &matrix::SparseMatrixCSR<T>,
    config: &matrix::config::MagnusConfig,
) -> matrix::SparseMatrixCSR<T>
where
    T: std::ops::AddAssign + Copy + num_traits::Num + Send + Sync,
{
    // Verify matrix dimensions
    assert_eq!(
        a.n_cols, b.n_rows,
        "Matrix dimensions must be compatible for multiplication"
    );

    // 1. Determine row categories for adaptive strategy selection
    let row_categories = matrix::categorize_rows(a, b, config);

    // 2. Allocate output matrix structures
    let n_rows = a.n_rows;
    let n_cols = b.n_cols;

    // 3. Process each row in parallel using the appropriate strategy
    let row_results: Vec<(Vec<usize>, Vec<T>)> = (0..n_rows)
        .into_par_iter() // Use Rayon's parallel iterator
        .map(|i| {
            let category = row_categories[i];

            match category {
                // For rows with small intermediate products, use sort accumulator
                matrix::config::RowCategory::Sort => accumulator::multiply_row_sort(i, a, b),

                // For rows where dense array fits in L2 cache, use dense accumulator
                matrix::config::RowCategory::DenseAccumulation => {
                    accumulator::multiply_row_dense(i, a, b)
                }

                // For rows requiring fine-level reordering
                matrix::config::RowCategory::FineLevel => {
                    reordering::multiply_row_fine_level(i, a, b, config)
                }

                // For rows with extremely large intermediate products
                matrix::config::RowCategory::CoarseLevel => {
                    reordering::multiply_row_coarse_level(i, a, b, config)
                }
            }
        })
        .collect();

    // 4. Assemble the final CSR matrix
    // We need to first compute the row pointers
    let mut row_ptr = Vec::with_capacity(n_rows + 1);
    row_ptr.push(0);

    let mut running_nnz = 0;
    for (cols, _) in &row_results {
        running_nnz += cols.len();
        row_ptr.push(running_nnz);
    }

    // Pre-allocate arrays for the combined result
    let mut col_idx = Vec::with_capacity(running_nnz);
    let mut values = Vec::with_capacity(running_nnz);

    // Combine all rows
    for (cols, vals) in row_results {
        col_idx.extend(cols);
        values.extend(vals);
    }

    // Create and return the final matrix
    matrix::SparseMatrixCSR::new(n_rows, n_cols, row_ptr, col_idx, values)
}

/// Process a batch of rows using coarse-level reordering with parallel chunk processing
///
/// This function implements a parallel version of coarse-level reordering (Algorithm 4),
/// processing multiple rows in batches and using parallel processing for chunk computations.
///
/// # Arguments
///
/// * `a` - Matrix A in CSR format
/// * `b` - Matrix B in CSR format
/// * `coarse_rows` - Array of row indices to process
/// * `config` - Configuration parameters
///
/// # Returns
///
/// A vector of tuples (row_idx, column_indices, values) for each processed row
pub fn process_coarse_level_rows_parallel<T>(
    a: &matrix::SparseMatrixCSR<T>,
    b: &matrix::SparseMatrixCSR<T>,
    coarse_rows: &[usize],
    config: &matrix::config::MagnusConfig,
) -> Vec<(usize, Vec<usize>, Vec<T>)>
where
    T: std::ops::AddAssign + Copy + num_traits::Num + Send + Sync + std::fmt::Debug,
{
    use std::sync::{Arc, Mutex};

    if coarse_rows.is_empty() {
        return Vec::new();
    }

    // Determine batch size based on config
    // Default to a reasonable value if not specified
    let batch_size = std::cmp::min(
        coarse_rows.len(),
        config
            .coarse_batch_size
            .unwrap_or(std::cmp::min(32, coarse_rows.len())),
    );

    // Create AHatCSC for the relevant rows
    let a_hat_csc = reordering::coarse::AHatCSC::new(a, coarse_rows);

    // Initialize result structures
    let mut results = Vec::with_capacity(coarse_rows.len());
    for &row_idx in coarse_rows {
        results.push((row_idx, Vec::new(), Vec::new()));
    }

    // Process in batches
    for batch_start in (0..coarse_rows.len()).step_by(batch_size) {
        let batch_end = std::cmp::min(batch_start + batch_size, coarse_rows.len());

        // Create fine-level reordering instance for this batch
        let reordering = reordering::fine::FineLevelReordering::new(b.n_cols, config);
        let metadata = reordering.get_metadata();
        let n_chunks = metadata.n_chunks;

        // Create shared mutex-protected result accumulators for this batch
        let batch_results: BatchResults<T> =
            Arc::new(Mutex::new(vec![Vec::new(); batch_end - batch_start]));

        // Process chunks in parallel
        (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
            // Calculate chunk boundaries
            let chunk_start = chunk_idx * metadata.chunk_length;
            let chunk_end = std::cmp::min(chunk_start + metadata.chunk_length, b.n_cols);

            if chunk_start == chunk_end {
                return; // Empty chunk
            }

            // Process this chunk for all rows in the batch
            let mut chunk_results = vec![Vec::<(usize, T)>::new(); batch_end - batch_start];
            process_column_chunk_parallel(
                &a_hat_csc,
                b,
                batch_start,
                batch_end,
                chunk_start,
                chunk_end,
                &mut chunk_results,
            );

            // Merge results back into the shared structure
            if !chunk_results.iter().all(|r| r.is_empty()) {
                let mut batch_accumulators = batch_results.lock().unwrap();
                for i in 0..(batch_end - batch_start) {
                    batch_accumulators[i].append(&mut chunk_results[i]);
                }
            }
        });

        // Extract and sort the results for this batch
        let batch_accumulators = Arc::try_unwrap(batch_results)
            .expect("Failed to reclaim batch results")
            .into_inner()
            .expect("Failed to unlock batch results");

        // Convert accumulated results to CSR format for each row
        for (batch_idx, accumulator) in batch_accumulators.into_iter().enumerate() {
            let result_idx = batch_start + batch_idx;

            if accumulator.is_empty() {
                continue;
            }

            // Sort by column index and combine duplicates
            let (col_indices, values) = sort_and_combine_entries(accumulator);

            // Store the results
            results[result_idx].1 = col_indices;
            results[result_idx].2 = values;
        }
    }

    results
}

/// Process a chunk of columns for parallel batch processing
///
/// This function is similar to the process_column_chunk method in CoarseLevelReordering,
/// but optimized for parallel execution.
fn process_column_chunk_parallel<T>(
    a_hat_csc: &reordering::coarse::AHatCSC<T>,
    b: &matrix::SparseMatrixCSR<T>,
    start_idx: usize,
    end_idx: usize,
    chunk_start: usize,
    chunk_end: usize,
    chunk_accumulators: &mut [Vec<(usize, T)>],
) where
    T: std::ops::AddAssign + Copy + num_traits::Num + Send + Sync,
{
    // Process each column in the chunk
    for col_idx in chunk_start..chunk_end {
        // Skip if column index is out of bounds (safety check)
        if col_idx >= a_hat_csc.matrix.col_ptr.len() - 1 || col_idx >= b.row_ptr.len() - 1 {
            continue;
        }

        // Get the elements in column col_idx of A
        let col_start = a_hat_csc.matrix.col_ptr[col_idx];
        let col_end = a_hat_csc.matrix.col_ptr[col_idx + 1];

        if col_start == col_end {
            continue; // Empty column
        }

        // Get the elements in row col_idx of B
        let b_row_start = b.row_ptr[col_idx];
        let b_row_end = b.row_ptr[col_idx + 1];

        if b_row_start == b_row_end {
            continue; // Empty row in B
        }

        // For each element in A[:, col_idx]
        for a_idx in col_start..col_end {
            let a_row = a_hat_csc.matrix.row_idx[a_idx];
            let a_val = a_hat_csc.matrix.values[a_idx];

            // Find the batch index for this row
            let batch_idx_opt = a_hat_csc.original_row_indices[start_idx..end_idx]
                .iter()
                .position(|&r| r == a_row);

            if let Some(batch_idx) = batch_idx_opt {
                // For each element in B[col_idx, :]
                for b_idx in b_row_start..b_row_end {
                    let b_col = b.col_idx[b_idx];
                    let b_val = b.values[b_idx];

                    let product = a_val * b_val;
                    chunk_accumulators[batch_idx].push((b_col, product));
                }
            }
        }
    }
}

/// Sort entries by column index and combine duplicates
///
/// This is a utility function for processing the accumulated entries
/// from coarse-level reordering.
fn sort_and_combine_entries<T>(entries: Vec<(usize, T)>) -> (Vec<usize>, Vec<T>)
where
    T: std::ops::AddAssign + Copy + num_traits::Num + std::fmt::Debug,
{
    if entries.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Sort by column index
    let mut sorted_entries = entries;
    sorted_entries.sort_by_key(|entry| entry.0);

    // Initialize result vectors
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    // Combine duplicates
    let mut current_col = sorted_entries[0].0;
    let mut current_val = sorted_entries[0].1;

    for entry in sorted_entries.iter().skip(1) {
        if entry.0 == current_col {
            // Same column, accumulate value
            current_val += entry.1;
        } else {
            // New column, store previous result
            col_indices.push(current_col);
            values.push(current_val);

            // Start new accumulation
            current_col = entry.0;
            current_val = entry.1;
        }
    }

    // Don't forget to add the last entry
    col_indices.push(current_col);
    values.push(current_val);

    (col_indices, values)
}
