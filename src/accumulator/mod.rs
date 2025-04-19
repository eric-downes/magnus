//! Accumulator implementations for intermediate products

// This is a placeholder for now.
// Will be implemented in Phase 1 of the roadmap.

/// Marker trait for accumulators
pub trait Accumulator<T> {
    /// Accumulates intermediate products with the given column indices and values
    fn accumulate(&self, col_indices: &[usize], values: &[T]) -> (Vec<usize>, Vec<T>);
}