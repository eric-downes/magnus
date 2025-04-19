//! # Parallel Implementation of MAGNUS SpGEMM
//!
//! This module provides parallel implementations of the MAGNUS algorithm
//! using Rayon for parallel processing of matrix rows.

use rayon::prelude::*;
use num_traits::Num;
use std::ops::AddAssign;

use crate::matrix;
use crate::accumulator;
use crate::reordering;

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
    assert_eq!(a.n_cols, b.n_rows, "Matrix dimensions must be compatible for multiplication");

    // 1. Determine row categories for adaptive strategy selection
    let row_categories = matrix::categorize_rows(a, b, config);
    
    // 2. Allocate output matrix structures
    let n_rows = a.n_rows;
    let n_cols = b.n_cols;
    
    // 3. Process each row in parallel using the appropriate strategy
    let row_results: Vec<(Vec<usize>, Vec<T>)> = (0..n_rows)
        .into_par_iter()  // Use Rayon's parallel iterator
        .map(|i| {
            let category = row_categories[i];
            
            match category {
                // For rows with small intermediate products, use sort accumulator
                matrix::config::RowCategory::Sort => {
                    accumulator::multiply_row_sort(i, a, b)
                },
                
                // For rows where dense array fits in L2 cache, use dense accumulator
                matrix::config::RowCategory::DenseAccumulation => {
                    accumulator::multiply_row_dense(i, a, b)
                },
                
                // For rows requiring fine-level reordering
                matrix::config::RowCategory::FineLevel => {
                    reordering::multiply_row_fine_level(i, a, b, config)
                },
                
                // For rows with extremely large intermediate products
                matrix::config::RowCategory::CoarseLevel => {
                    reordering::multiply_row_coarse_level(i, a, b, config)
                },
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

// Batch processing of coarse-level rows
//
// This parallel version was mentioned in the roadmap but 
// may require additional data structure changes.
// Leaving as TODO for future optimization.
pub fn process_coarse_level_rows_parallel<T>(
    a: &matrix::SparseMatrixCSR<T>,
    b: &matrix::SparseMatrixCSR<T>,
    coarse_rows: &[usize],
    config: &matrix::config::MagnusConfig,
) -> Vec<(usize, Vec<usize>, Vec<T>)>
where
    T: std::ops::AddAssign + Copy + num_traits::Num + Send + Sync,
{
    // For now, call the serial version
    // This is a placeholder for future optimization
    let results = reordering::process_coarse_level_rows(a, b, coarse_rows, config);
    
    // Convert to the expected format
    coarse_rows.iter()
        .zip(results.into_iter())
        .map(|(&row_idx, (col_indices, values))| (row_idx, col_indices, values))
        .collect()
}