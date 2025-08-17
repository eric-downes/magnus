//! # MAGNUS: Matrix Algebra for GPU and Multicore Systems
//!
//! MAGNUS is a high-performance algorithm for multiplying large sparse matrices,
//! as described in [this paper](https://arxiv.org/pdf/2501.07056).
//!
//! ## Overview
//!
//! This library implements the MAGNUS algorithm in Rust, with a focus on:
//!
//! - Hardware-agnostic implementation with architecture-specific optimizations
//! - Adaptive approach that chooses different strategies based on row properties
//! - Memory-efficient operations through improved data locality
//! - High performance through vectorization and parallelization
//!
//! ## Algorithm Components
//!
//! The MAGNUS algorithm consists of several key components:
//!
//! 1. **Row Categorization**: Analyzing the computational requirements of each row
//!    and categorizing it into one of four categories.
//!
//! 2. **Accumulation Methods**:
//!    - **Sort-based**: For small intermediate products
//!    - **Dense**: For intermediate products that fit in L2 cache
//!
//! 3. **Reordering Strategies**:
//!    - **Fine-level**: For larger intermediate products
//!    - **Coarse-level**: For extremely large intermediate products
//!
//! ## Usage
//!
//! Basic matrix multiplication:
//!
//! ```
//! use magnus::{SparseMatrixCSR, MagnusConfig, magnus_spgemm};
//!
//! // Create matrices (implementation details omitted)
//! # let a = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
//! # let b = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
//! let config = MagnusConfig::default();
//!
//! // Multiply matrices using MAGNUS
//! let c = magnus_spgemm(&a, &b, &config);
//! ```
//!
//! For parallel execution on multi-core processors:
//!
//! ```
//! use magnus::{SparseMatrixCSR, MagnusConfig, magnus_spgemm_parallel};
//!
//! # let a = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
//! # let b = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
//! let config = MagnusConfig::default();
//!
//! // Multiply matrices using parallel MAGNUS
//! let c = magnus_spgemm_parallel(&a, &b, &config);
//! ```
//!
//! A reference implementation is also available for testing and comparison:
//!
//! ```
//! use magnus::{SparseMatrixCSR, reference_spgemm};
//!
//! # let a = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
//! # let b = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
//! let c = reference_spgemm(&a, &b);
//! ```

pub mod accumulator;
pub mod constants;
pub mod matrix;
pub mod parallel;
pub mod reordering;
pub mod utils;

// Re-export primary components
pub use accumulator::{
    create_accumulator, create_simd_accelerator, create_simd_accelerator_f32, multiply_row_dense,
    multiply_row_sort, Accumulator, FallbackAccumulator, SimdAccelerator,
};
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use accumulator::{AccelerateAccumulator, NeonAccumulator};
pub use matrix::config::{
    detect_architecture, Architecture, MagnusConfig, RowCategory, SortMethod, SystemParameters,
};
pub use matrix::{analyze_categorization, categorize_rows, CategorizationSummary};
pub use matrix::{reference_spgemm, SparseMatrixCSC, SparseMatrixCSR};
pub use parallel::{magnus_spgemm_parallel, process_coarse_level_rows_parallel};
pub use reordering::{
    multiply_row_coarse_level, multiply_row_fine_level, process_coarse_level_rows,
};
pub use utils::{from_sprs_csc, from_sprs_csr, to_sprs_csc, to_sprs_csr};

/// Performs sparse general matrix-matrix multiplication (SpGEMM)
/// using the MAGNUS algorithm.
///
/// This is the main entry point for the library. It implements the full MAGNUS
/// algorithm, including row categorization and adaptive strategy selection.
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
/// use magnus::{SparseMatrixCSR, magnus_spgemm, MagnusConfig};
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
/// // Multiply the matrices using MAGNUS
/// let config = MagnusConfig::default();
/// let c = magnus_spgemm(&a, &b, &config);
///
/// // Result should be a diagonal matrix with elements 2.0
/// assert_eq!(c.n_rows, 2);
/// assert_eq!(c.n_cols, 2);
/// assert_eq!(c.row_ptr, vec![0, 1, 2]);
/// assert_eq!(c.col_idx, vec![0, 1]);
/// assert!((c.values[0] - 2.0).abs() < 1e-10);
/// assert!((c.values[1] - 2.0).abs() < 1e-10);
/// ```
pub fn magnus_spgemm<T>(
    a: &matrix::SparseMatrixCSR<T>,
    b: &matrix::SparseMatrixCSR<T>,
    config: &matrix::config::MagnusConfig,
) -> matrix::SparseMatrixCSR<T>
where
    T: std::ops::AddAssign + Copy + num_traits::Num,
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

    // We'll first collect the results for each row
    let mut row_results: Vec<(Vec<usize>, Vec<T>)> = Vec::with_capacity(n_rows);

    // 3. Process each row using the appropriate strategy
    for i in 0..n_rows {
        let category = row_categories[i];

        let (col_indices, values) = match category {
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
        };

        row_results.push((col_indices, values));
    }

    // 4. Assemble the final CSR matrix
    let mut row_ptr = Vec::with_capacity(n_rows + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    // Calculate row pointers and combined nnz
    row_ptr.push(0);
    let mut nnz = 0;

    for (cols, _) in &row_results {
        nnz += cols.len();
        row_ptr.push(nnz);
    }

    // Allocate arrays for combined result
    col_idx.reserve(nnz);
    values.reserve(nnz);

    // Combine all rows
    for (cols, vals) in row_results {
        col_idx.extend(cols);
        values.extend(vals);
    }

    // Create and return the final matrix
    matrix::SparseMatrixCSR::new(n_rows, n_cols, row_ptr, col_idx, values)
}

/// Version information for the MAGNUS library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
