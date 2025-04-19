//! Reference implementation of SpGEMM using manual multiplication
//!
//! This provides a baseline for correctness testing and performance comparison.
//! We implement a simple but correct SpGEMM algorithm to use as reference.

use num_traits::Num;
use std::collections::HashMap;
use std::ops::AddAssign;

use crate::matrix::SparseMatrixCSR;

/// Performs sparse matrix multiplication using a simple algorithm as a reference implementation
///
/// This implementation uses a simple row-by-row approach with a hashmap accumulator.
/// It's not optimized for performance but provides a correct reference result.
pub fn reference_spgemm<T>(a: &SparseMatrixCSR<T>, b: &SparseMatrixCSR<T>) -> SparseMatrixCSR<T>
where
    T: Copy + Num + AddAssign,
{
    assert_eq!(
        a.n_cols, b.n_rows,
        "Matrix dimensions must be compatible for multiplication"
    );

    let n_rows = a.n_rows;
    let n_cols = b.n_cols;

    // Prepare output CSR data structures
    let mut row_ptr = Vec::with_capacity(n_rows + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    // Push initial row pointer
    row_ptr.push(0);

    // Process each row of A
    for i in 0..n_rows {
        // Use a hashmap as an accumulator for this row
        let mut accum: HashMap<usize, T> = HashMap::new();

        // For each non-zero in row i of A
        for (k, &a_val) in a.row_iter(i) {
            // For each non-zero in row k of B
            let b_row_start = b.row_ptr[k];
            let b_row_end = b.row_ptr[k + 1];

            for b_idx in b_row_start..b_row_end {
                let j = b.col_idx[b_idx];
                let b_val = b.values[b_idx];

                // Multiply and accumulate
                let product = a_val * b_val;
                *accum.entry(j).or_insert(T::zero()) += product;
            }
        }

        // Convert hashmap to sorted (col_idx, values) pairs
        let mut row_entries: Vec<_> = accum.into_iter().collect();
        row_entries.sort_by_key(|&(col, _)| col);

        // Append to CSR data structures
        for (j, val) in row_entries {
            if !val.is_zero() {
                // Skip zeros
                col_idx.push(j);
                values.push(val);
            }
        }

        // Record end of this row
        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(n_rows, n_cols, row_ptr, col_idx, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_multiplication() {
        // Create test matrices:
        // A = [1 2; 0 3]
        // B = [4 5; 6 7]
        // Expected result: C = A*B = [16 19; 18 21]

        let a = SparseMatrixCSR::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1, 2, 3]);

        let b = SparseMatrixCSR::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![4, 5, 6, 7]);

        let result = reference_spgemm(&a, &b);

        assert_eq!(result.n_rows, 2);
        assert_eq!(result.n_cols, 2);
        assert_eq!(result.nnz(), 4);

        // Convert result to a dense representation for easier verification
        let mut dense_result = vec![vec![0; 2]; 2];
        for i in 0..2 {
            for (j, &val) in result.row_iter(i) {
                dense_result[i][j] = val;
            }
        }

        assert_eq!(dense_result[0][0], 16);
        assert_eq!(dense_result[0][1], 19);
        assert_eq!(dense_result[1][0], 18);
        assert_eq!(dense_result[1][1], 21);
    }

    #[test]
    fn test_identity_multiplication() {
        // Create identity and diagonal matrices
        let identity = SparseMatrixCSR::<i32>::identity(3);

        let diagonal = SparseMatrixCSR::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![5, 6, 7]);

        // Identity * Diagonal should equal Diagonal
        let result = reference_spgemm(&identity, &diagonal);

        assert_eq!(result.n_rows, 3);
        assert_eq!(result.n_cols, 3);
        assert_eq!(result.nnz(), 3);

        // Check each row
        for i in 0..3 {
            let row: Vec<_> = result.row_iter(i).collect();
            assert_eq!(row.len(), 1);
            assert_eq!(row[0].0, i);
            assert_eq!(*row[0].1, i as i32 + 5);
        }
    }
}
