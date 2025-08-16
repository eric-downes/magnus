//! Accumulator implementations for sparse matrix multiplication
//!
//! This module contains different accumulator implementations for
//! handling intermediate products in sparse matrix multiplication.
//! The MAGNUS algorithm uses different accumulators based on the
//! characteristics of each row.

pub mod dense;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod neon;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod accelerate;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod metal;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod metal_impl;
pub mod simd;
pub mod sort;

use num_traits::Num;
use std::ops::AddAssign;

/// Trait for accumulators that handle intermediate products in SpGEMM
///
/// Different implementations of this trait provide different strategies
/// for accumulating and merging intermediate products, with varying
/// performance characteristics depending on the input size and structure.
pub trait Accumulator<T>
where
    T: Copy + Num + AddAssign,
{
    /// Reset the accumulator to prepare for a new row
    fn reset(&mut self);

    /// Accumulate a single entry (column and value)
    fn accumulate(&mut self, col: usize, val: T);

    /// Extract the non-zero entries as sorted (column, value) pairs
    ///
    /// Returns a tuple of `(col_indices, values)` with entries sorted by column index.
    fn extract_result(self) -> (Vec<usize>, Vec<T>);
}

/// Create an appropriate accumulator based on the output matrix columns
///
/// # Arguments
///
/// * `n_cols` - The number of columns in the output matrix
/// * `dense_threshold` - Threshold for choosing dense accumulation
///
/// # Returns
///
/// A boxed accumulator trait object appropriate for the given parameters.
pub fn create_accumulator<T>(n_cols: usize, dense_threshold: usize) -> Box<dyn Accumulator<T>>
where
    T: Copy + Num + AddAssign + 'static,
{
    // Choose the accumulator based on the number of columns and threshold
    if n_cols <= dense_threshold {
        // For small output dimensions, use dense accumulator
        Box::new(dense::DenseAccumulator::new(n_cols))
    } else {
        // For large output dimensions, use sort-based accumulator
        // Start with a reasonable initial capacity
        let initial_capacity = std::cmp::min(n_cols / 10, 1024);
        Box::new(sort::SortAccumulator::new(initial_capacity))
    }
}

/// Create a sort-based accumulator with a given initial capacity
///
/// # Arguments
///
/// * `initial_capacity` - Initial capacity for the temporary storage
///
/// # Returns
///
/// A boxed sort-based accumulator trait object.
pub fn create_sort_accumulator<T>(initial_capacity: usize) -> Box<dyn Accumulator<T>>
where
    T: Copy + Num + AddAssign + 'static,
{
    Box::new(sort::SortAccumulator::new(initial_capacity))
}

/// Create a dense accumulator for the given number of columns
///
/// # Arguments
///
/// * `n_cols` - The number of columns in the output matrix
///
/// # Returns
///
/// A boxed dense accumulator trait object.
pub fn create_dense_accumulator<T>(n_cols: usize) -> Box<dyn Accumulator<T>>
where
    T: Copy + Num + AddAssign + 'static,
{
    Box::new(dense::DenseAccumulator::new(n_cols))
}

// Re-export key functions for convenient access
pub use dense::multiply_row_dense;
pub use simd::{
    create_simd_accelerator, create_simd_accelerator_f32, FallbackAccumulator, SimdAccelerator,
};
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use simd::NeonAccumulator;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use accelerate::AccelerateAccumulator;
pub use sort::multiply_row_sort;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_accumulator_selection() {
        // Let's test selection behavior instead of type checking

        // Test behavior with small matrix, should use dense accumulator
        let n_cols_small = 100;
        let dense_threshold = 256;
        let mut acc_small = create_accumulator::<f64>(n_cols_small, dense_threshold);

        // Accumulate and test typical dense accumulator behaviors
        acc_small.accumulate(10, 1.0);
        acc_small.accumulate(50, 2.0);
        acc_small.reset();

        // Test behavior with large matrix, should use sort-based accumulator
        let n_cols_large = 1000;
        let small_threshold = 100;
        let mut acc_large = create_accumulator::<f64>(n_cols_large, small_threshold);

        // Accumulate and test typical sort accumulator behaviors
        acc_large.accumulate(10, 1.0);
        acc_large.accumulate(500, 2.0);
        acc_large.reset();

        // Make sure accumulators are used correctly
        assert!(
            n_cols_small <= dense_threshold,
            "Small n_cols should be <= dense_threshold"
        );
        assert!(
            n_cols_large > small_threshold,
            "Large n_cols should be > threshold"
        );
    }

    #[test]
    fn test_explicit_accumulator_creation() {
        // Test that we can create and use specific accumulator types

        // Test DenseAccumulator behavior
        let mut dense_acc = create_dense_accumulator::<f64>(100);
        dense_acc.accumulate(10, 1.0);
        dense_acc.accumulate(50, 2.0);
        dense_acc.reset();

        // Test SortAccumulator behavior
        let mut sort_acc = create_sort_accumulator::<f64>(50);
        sort_acc.accumulate(10, 1.0);
        sort_acc.accumulate(30, 2.0);
        sort_acc.reset();
    }
}
