//! Dense accumulator implementation for SpGEMM
//!
//! This module implements the dense accumulation method (Algorithm 1 in the paper)
//! for multiplying sparse matrices.

use num_traits::Num;
use std::ops::AddAssign;

use crate::accumulator::Accumulator;
use crate::matrix::SparseMatrixCSR;

/// Dense accumulator for a single row of sparse matrix multiplication
///
/// This implements Algorithm 1 from the MAGNUS paper, which uses a dense
/// array to accumulate results when the intermediate products are expected
/// to have relatively high density or when the dense array fits in cache.
pub struct DenseAccumulator<T> {
    /// The dense accumulation array
    values: Vec<T>,

    /// Flags to track which positions in the dense array are non-zero
    occupied: Vec<bool>,

    /// Temporary storage for the column indices of non-zero elements
    col_indices: Vec<usize>,
}

impl<T> DenseAccumulator<T>
where
    T: Copy + Num + AddAssign,
{
    /// Create a new dense accumulator with specified column capacity
    ///
    /// # Arguments
    ///
    /// * `n_cols` - The number of columns in the output matrix (C)
    pub fn new(n_cols: usize) -> Self {
        Self {
            values: vec![T::zero(); n_cols],
            occupied: vec![false; n_cols],
            col_indices: Vec::new(),
        }
    }

    /// Reset the accumulator for reuse without reallocating memory
    fn reset(&mut self) {
        // Clear the occupied flags
        for &col in &self.col_indices {
            self.occupied[col] = false;
        }

        // Clear the non-zero indices list
        self.col_indices.clear();
    }

    /// Accumulate a single entry (column and value)
    ///
    /// # Arguments
    ///
    /// * `col` - The column index
    /// * `val` - The value to accumulate
    fn accumulate(&mut self, col: usize, val: T) {
        if !self.occupied[col] {
            // First time seeing this column, mark it and add to indices
            self.occupied[col] = true;
            self.col_indices.push(col);
            self.values[col] = val;
        } else {
            // Column already exists, just add the value
            self.values[col] += val;
        }
    }

    /// Extract the non-zero entries as sorted (column, value) pairs
    ///
    /// Returns a tuple of `(col_indices, values)` with entries sorted by column index.
    fn extract_result(mut self) -> (Vec<usize>, Vec<T>) {
        // Sort the column indices
        self.col_indices.sort_unstable();

        // Extract the values in column order
        let values = self
            .col_indices
            .iter()
            .map(|&col| self.values[col])
            .collect();

        (self.col_indices, values)
    }
}

impl<T> Accumulator<T> for DenseAccumulator<T>
where
    T: Copy + Num + AddAssign,
{
    fn reset(&mut self) {
        DenseAccumulator::reset(self)
    }

    fn accumulate(&mut self, col: usize, val: T) {
        DenseAccumulator::accumulate(self, col, val)
    }

    fn extract_result(self) -> (Vec<usize>, Vec<T>) {
        DenseAccumulator::extract_result(self)
    }
}

/// Multiply a single row of matrix A with matrix B using a dense accumulator
///
/// # Arguments
///
/// * `a_row` - Row index in matrix A to multiply
/// * `a` - Matrix A in CSR format
/// * `b` - Matrix B in CSR format
///
/// # Returns
///
/// A tuple of `(col_indices, values)` containing the non-zero entries of the result row.
pub fn multiply_row_dense<T>(
    a_row: usize,
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
) -> (Vec<usize>, Vec<T>)
where
    T: Copy + Num + AddAssign,
{
    // Create a dense accumulator with capacity for all columns of B
    let mut accumulator = DenseAccumulator::new(b.n_cols);

    // For each non-zero in row a_row of A
    let row_start = a.row_ptr[a_row];
    let row_end = a.row_ptr[a_row + 1];

    for a_idx in row_start..row_end {
        let b_row = a.col_idx[a_idx]; // This is the row of B we need to process
        let a_val = a.values[a_idx]; // Value from A

        // Process row b_row of matrix B
        let b_row_start = b.row_ptr[b_row];
        let b_row_end = b.row_ptr[b_row + 1];

        for b_idx in b_row_start..b_row_end {
            let b_col = b.col_idx[b_idx]; // Column in B (and result)
            let b_val = b.values[b_idx]; // Value from B

            // Multiply and accumulate
            let product = a_val * b_val;
            accumulator.accumulate(b_col, product);
        }
    }

    // Extract the result
    accumulator.extract_result()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_accumulator_empty() {
        let accumulator = DenseAccumulator::<f64>::new(5);

        // Without any accumulation, should get empty result
        let (col_indices, values) = accumulator.extract_result();

        assert_eq!(col_indices.len(), 0);
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_dense_accumulator_single_entry() {
        let mut accumulator = DenseAccumulator::<f64>::new(5);

        // Add a single value
        accumulator.accumulate(2, 3.5f64);

        // Extract the result
        let (col_indices, values) = accumulator.extract_result();

        assert_eq!(col_indices, vec![2]);
        let expected: f64 = 3.5;
        let diff: f64 = (values[0] - expected).abs();
        assert!(diff < 1.0e-10);
    }

    #[test]
    fn test_dense_accumulator_multiple_entries() {
        let mut accumulator = DenseAccumulator::<f64>::new(5);

        // Add multiple values in different columns
        accumulator.accumulate(1, 2.0f64);
        accumulator.accumulate(3, 4.0f64);
        accumulator.accumulate(0, 1.0f64);

        // Extract the result
        let (col_indices, values) = accumulator.extract_result();

        // Result should be sorted by column index
        assert_eq!(col_indices, vec![0, 1, 3]);

        // Test each value individually with explicit type annotations
        let expected_vals = [1.0f64, 2.0f64, 4.0f64];
        for i in 0..3 {
            let diff: f64 = (values[i] - expected_vals[i]).abs();
            assert!(diff < 1.0e-10);
        }
    }

    #[test]
    fn test_dense_accumulator_duplicate_columns() {
        let mut accumulator = DenseAccumulator::<f64>::new(5);

        // Add multiple values to the same column
        accumulator.accumulate(2, 1.5f64);
        accumulator.accumulate(2, 2.5f64);
        accumulator.accumulate(2, 1.0f64);

        // Extract the result
        let (col_indices, values) = accumulator.extract_result();

        // Should have one entry with the sum of all values
        assert_eq!(col_indices, vec![2]);
        let expected: f64 = 5.0;
        let diff: f64 = (values[0] - expected).abs();
        assert!(diff < 1.0e-10);
    }

    #[test]
    fn test_dense_accumulator_reset() {
        let mut accumulator = DenseAccumulator::<f64>::new(5);

        // Add some values
        accumulator.accumulate(1, 2.0f64);
        accumulator.accumulate(3, 4.0f64);

        // Reset the accumulator
        accumulator.reset();

        // Add new values
        accumulator.accumulate(0, 1.0f64);
        accumulator.accumulate(4, 5.0f64);

        // Extract the result
        let (col_indices, values) = accumulator.extract_result();

        // Only the new values should be present
        assert_eq!(col_indices, vec![0, 4]);

        // Test each value individually with explicit type annotations
        let expected_vals = [1.0f64, 5.0f64];
        for i in 0..2 {
            let diff: f64 = (values[i] - expected_vals[i]).abs();
            assert!(diff < 1.0e-10);
        }
    }

    #[test]
    fn test_multiply_row_dense() {
        // Create test matrices:
        // A = [1 2 0; 0 3 4; 5 0 6]
        // B = [7 0 0; 0 8 0; 0 0 9]
        // Expected result row 0: [7 16 0]
        // Expected result row 1: [0 24 36]
        // Expected result row 2: [35 0 54]

        let a = SparseMatrixCSR::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        let b = SparseMatrixCSR::new(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![7.0f64, 8.0, 9.0],
        );

        // Multiply row 0 of A with B
        let (cols_row0, vals_row0) = multiply_row_dense(0, &a, &b);

        // Should have 2 non-zeros: (0, 7) and (1, 16)
        assert_eq!(cols_row0, vec![0, 1]);
        let diff1: f64 = (vals_row0[0] - 7.0f64).abs();
        let diff2: f64 = (vals_row0[1] - 16.0f64).abs();
        assert!(diff1 < 1.0e-10);
        assert!(diff2 < 1.0e-10);

        // Multiply row 1 of A with B
        let (cols_row1, vals_row1) = multiply_row_dense(1, &a, &b);

        // Should have 2 non-zeros: (1, 24) and (2, 36)
        assert_eq!(cols_row1, vec![1, 2]);
        let diff3: f64 = (vals_row1[0] - 24.0f64).abs();
        let diff4: f64 = (vals_row1[1] - 36.0f64).abs();
        assert!(diff3 < 1.0e-10);
        assert!(diff4 < 1.0e-10);

        // Multiply row 2 of A with B
        let (cols_row2, vals_row2) = multiply_row_dense(2, &a, &b);

        // Should have 2 non-zeros: (0, 35) and (2, 54)
        assert_eq!(cols_row2, vec![0, 2]);
        let diff5: f64 = (vals_row2[0] - 35.0f64).abs();
        let diff6: f64 = (vals_row2[1] - 54.0f64).abs();
        assert!(diff5 < 1.0e-10);
        assert!(diff6 < 1.0e-10);
    }
}
