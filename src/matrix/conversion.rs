//! Conversion functions between matrix formats

use crate::matrix::{SparseMatrixCSC, SparseMatrixCSR};
use num_traits::Num;

impl<T: Copy + Num> SparseMatrixCSR<T> {
    /// Converts this CSR matrix to CSC format
    pub fn to_csc(&self) -> SparseMatrixCSC<T> {
        // Count non-zeros per column
        let mut col_counts = vec![0; self.n_cols];

        for &col in &self.col_idx {
            col_counts[col] += 1;
        }

        // Compute column pointers via prefix sum
        let mut col_ptr = vec![0; self.n_cols + 1];
        let mut sum = 0;

        for (i, &count) in col_counts.iter().enumerate() {
            col_ptr[i] = sum;
            sum += count;
        }
        col_ptr[self.n_cols] = sum;

        // Allocate arrays for CSC matrix
        let nnz = self.nnz();
        let mut row_idx = vec![0; nnz];
        let mut values = vec![T::zero(); nnz];

        // Fill CSC matrix
        let mut temp_col_ptr = col_ptr.clone();

        for i in 0..self.n_rows {
            let row_start = self.row_ptr[i];
            let row_end = self.row_ptr[i + 1];

            for idx in row_start..row_end {
                let col = self.col_idx[idx];
                let pos = temp_col_ptr[col];

                row_idx[pos] = i;
                values[pos] = self.values[idx];

                temp_col_ptr[col] += 1;
            }
        }

        SparseMatrixCSC::new(self.n_rows, self.n_cols, col_ptr, row_idx, values)
    }
}

impl<T: Copy + Num> SparseMatrixCSC<T> {
    /// Converts this CSC matrix to CSR format
    pub fn to_csr(&self) -> SparseMatrixCSR<T> {
        // Count non-zeros per row
        let mut row_counts = vec![0; self.n_rows];

        for &row in &self.row_idx {
            row_counts[row] += 1;
        }

        // Compute row pointers via prefix sum
        let mut row_ptr = vec![0; self.n_rows + 1];
        let mut sum = 0;

        for (i, &count) in row_counts.iter().enumerate() {
            row_ptr[i] = sum;
            sum += count;
        }
        row_ptr[self.n_rows] = sum;

        // Allocate arrays for CSR matrix
        let nnz = self.nnz();
        let mut col_idx = vec![0; nnz];
        let mut values = vec![T::zero(); nnz];

        // Fill CSR matrix
        let mut temp_row_ptr = row_ptr.clone();

        for j in 0..self.n_cols {
            let col_start = self.col_ptr[j];
            let col_end = self.col_ptr[j + 1];

            for idx in col_start..col_end {
                let row = self.row_idx[idx];
                let pos = temp_row_ptr[row];

                col_idx[pos] = j;
                values[pos] = self.values[idx];

                temp_row_ptr[row] += 1;
            }
        }

        SparseMatrixCSR::new(self.n_rows, self.n_cols, row_ptr, col_idx, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_to_csc_conversion() {
        // Create a CSR matrix
        //    [1 2 0]
        //    [0 3 0]
        //    [4 0 5]
        let csr = SparseMatrixCSR::new(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 0, 2],
            vec![1, 2, 3, 4, 5],
        );

        // Convert to CSC
        let csc = csr.to_csc();

        // Verify dimensions and nnz
        assert_eq!(csc.n_rows, 3);
        assert_eq!(csc.n_cols, 3);
        assert_eq!(csc.nnz(), 5);

        // Verify column pointers
        assert_eq!(csc.col_ptr, vec![0, 2, 4, 5]);

        // Check contents of first column
        let col0: Vec<_> = csc.col_iter(0).collect();
        assert_eq!(col0, vec![(0, &1), (2, &4)]);

        // Check contents of second column
        let col1: Vec<_> = csc.col_iter(1).collect();
        assert_eq!(col1, vec![(0, &2), (1, &3)]);

        // Check contents of third column
        let col2: Vec<_> = csc.col_iter(2).collect();
        assert_eq!(col2, vec![(2, &5)]);
    }

    #[test]
    fn test_csc_to_csr_conversion() {
        // Create a CSC matrix
        //    [1 2 0]
        //    [0 3 0]
        //    [4 0 5]
        let csc = SparseMatrixCSC::new(
            3,
            3,
            vec![0, 2, 4, 5],
            vec![0, 2, 0, 1, 2],
            vec![1, 4, 2, 3, 5],
        );

        // Convert to CSR
        let csr = csc.to_csr();

        // Verify dimensions and nnz
        assert_eq!(csr.n_rows, 3);
        assert_eq!(csr.n_cols, 3);
        assert_eq!(csr.nnz(), 5);

        // Verify row pointers
        assert_eq!(csr.row_ptr, vec![0, 2, 3, 5]);

        // Check contents of first row
        let row0: Vec<_> = csr.row_iter(0).collect();
        assert_eq!(row0, vec![(0, &1), (1, &2)]);

        // Check contents of second row
        let row1: Vec<_> = csr.row_iter(1).collect();
        assert_eq!(row1, vec![(1, &3)]);

        // Check contents of third row
        let row2: Vec<_> = csr.row_iter(2).collect();
        assert_eq!(row2, vec![(0, &4), (2, &5)]);
    }

    #[test]
    fn test_roundtrip_conversion() {
        // Create a CSR matrix
        let original_csr = SparseMatrixCSR::new(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 0, 2],
            vec![1, 2, 3, 4, 5],
        );

        // Convert to CSC and back to CSR
        let csc = original_csr.to_csc();
        let roundtrip_csr = csc.to_csr();

        // Verify dimensions and nnz
        assert_eq!(roundtrip_csr.n_rows, original_csr.n_rows);
        assert_eq!(roundtrip_csr.n_cols, original_csr.n_cols);
        assert_eq!(roundtrip_csr.nnz(), original_csr.nnz());

        // Verify row pointers
        assert_eq!(roundtrip_csr.row_ptr, original_csr.row_ptr);

        // The column indices and values might be in a different order within each row,
        // so we need to check row by row and sort the elements
        for i in 0..original_csr.n_rows {
            let mut original_row: Vec<_> = original_csr
                .row_iter(i)
                .map(|(col, &val)| (col, val))
                .collect();

            let mut roundtrip_row: Vec<_> = roundtrip_csr
                .row_iter(i)
                .map(|(col, &val)| (col, val))
                .collect();

            // Sort by column index for comparison
            original_row.sort_by_key(|&(col, _)| col);
            roundtrip_row.sort_by_key(|&(col, _)| col);

            assert_eq!(original_row, roundtrip_row);
        }
    }
}
