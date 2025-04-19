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
//! // let c = magnus_spgemm(&a, &b, &config); // Currently in progress
//! ```
//!
//! For now, a reference implementation is available:
//!
//! ```
//! use magnus::{SparseMatrixCSR, reference_spgemm};
//!
//! # let a = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
//! # let b = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
//! let c = reference_spgemm(&a, &b);
//! ```

pub mod matrix;
pub mod accumulator;
pub mod reordering;
pub mod utils;

// Re-export primary components
pub use matrix::{SparseMatrixCSR, SparseMatrixCSC, reference_spgemm};
pub use matrix::{categorize_rows, analyze_categorization, CategorizationSummary};
pub use matrix::config::{MagnusConfig, SystemParameters, Architecture, RowCategory, SortMethod};
pub use utils::{to_sprs_csr, to_sprs_csc, from_sprs_csr, from_sprs_csc};
pub use accumulator::{Accumulator, create_accumulator, multiply_row_dense, multiply_row_sort};

/// Performs sparse general matrix-matrix multiplication (SpGEMM)
/// using the MAGNUS algorithm.
///
/// This is the main entry point for the library.
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
/// // Example will be added when implementation is complete
/// # fn test_without_calling() {
/// #     let a = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
/// #     let b = SparseMatrixCSR::<f64>::new(1, 1, vec![0, 0], vec![], vec![]);
/// #     let config = MagnusConfig::default();
/// #     // Uncomment when implemented: let c = magnus_spgemm(&a, &b, &config);
/// # }
/// ```
pub fn magnus_spgemm<T>(
    _a: &matrix::SparseMatrixCSR<T>,
    _b: &matrix::SparseMatrixCSR<T>,
    _config: &matrix::config::MagnusConfig,
) -> matrix::SparseMatrixCSR<T> 
where
    T: std::ops::AddAssign + Copy + num_traits::Num,
{
    // Placeholder until implementation is complete
    unimplemented!("MAGNUS SpGEMM implementation is in progress")
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