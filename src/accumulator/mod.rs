//! Accumulator implementations for sparse matrix multiplication
//!
//! This module contains different accumulator implementations for
//! handling intermediate products in sparse matrix multiplication.
//! The MAGNUS algorithm uses different accumulators based on the
//! characteristics of each row.

pub mod dense;

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
pub fn create_accumulator<T>(n_cols: usize, _dense_threshold: usize) -> Box<dyn Accumulator<T>>
where
    T: Copy + Num + AddAssign + 'static,
{
    // For now, always use dense accumulator
    // In the future, we'll implement the sort-based accumulator too
    Box::new(dense::DenseAccumulator::new(n_cols))
}

// Re-export key functions for convenient access
pub use dense::multiply_row_dense;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_accumulator() {
        // Just test that we can create an accumulator for now
        let n_cols = 100;
        let dense_threshold = 256; // Default from paper
        let _acc = create_accumulator::<f64>(n_cols, dense_threshold);
        
        // More comprehensive tests will be added when more accumulator
        // types are implemented and the selection logic is in place
    }
}