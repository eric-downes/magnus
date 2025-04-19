//! Compressed Sparse Row (CSR) matrix format implementation

use std::fmt;
use num_traits::Num;

/// A sparse matrix in Compressed Sparse Row (CSR) format
///
/// The CSR format stores a sparse matrix using three arrays:
/// - row_ptr: Array of size n_rows + 1 containing indices into col_idx and values arrays
/// - col_idx: Array of size nnz containing column indices of non-zero elements
/// - values: Array of size nnz containing the non-zero values
///
/// This implementation is designed for the MAGNUS sparse matrix multiplication algorithm.
#[derive(Clone)]
pub struct SparseMatrixCSR<T> {
    /// Number of rows in the matrix
    pub n_rows: usize,
    
    /// Number of columns in the matrix
    pub n_cols: usize,
    
    /// Row pointers (size: n_rows + 1)
    /// row_ptr[i] is the index in col_idx and values where row i starts
    /// row_ptr[n_rows] is equal to nnz
    pub row_ptr: Vec<usize>,
    
    /// Column indices (size: nnz)
    pub col_idx: Vec<usize>,
    
    /// Non-zero values (size: nnz)
    pub values: Vec<T>,
}

impl<T> SparseMatrixCSR<T>
where
    T: Copy + Num,
{
    /// Creates a new CSR matrix with the given dimensions and data
    ///
    /// # Arguments
    ///
    /// * `n_rows` - Number of rows
    /// * `n_cols` - Number of columns
    /// * `row_ptr` - Row pointers
    /// * `col_idx` - Column indices
    /// * `values` - Non-zero values
    ///
    /// # Panics
    ///
    /// Panics if the input arrays are inconsistent:
    /// - row_ptr.len() must be n_rows + 1
    /// - col_idx.len() must equal values.len()
    /// - row_ptr[n_rows] must equal col_idx.len()
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        assert_eq!(row_ptr.len(), n_rows + 1, "row_ptr.len() must be n_rows + 1");
        assert_eq!(col_idx.len(), values.len(), "col_idx.len() must equal values.len()");
        assert_eq!(
            row_ptr[n_rows], col_idx.len(),
            "row_ptr[n_rows] must equal col_idx.len()"
        );
        
        // Check that column indices are within bounds
        for &col in &col_idx {
            assert!(col < n_cols, "Column index {} out of bounds (n_cols = {})", col, n_cols);
        }
        
        Self {
            n_rows,
            n_cols,
            row_ptr,
            col_idx,
            values,
        }
    }
    
    /// Returns the number of non-zero elements in the matrix
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Returns an iterator over the non-zero elements in row i
    ///
    /// Each item is a tuple (col_idx, value) representing a non-zero element
    pub fn row_iter(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        assert!(i < self.n_rows, "Row index out of bounds");
        
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        
        self.col_idx[start..end]
            .iter()
            .zip(&self.values[start..end])
            .map(|(&col, val)| (col, val))
    }
    
    /// Creates an empty matrix with the given dimensions
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        let row_ptr = vec![0; n_rows + 1];
        let col_idx = Vec::new();
        let values = Vec::new();
        
        Self {
            n_rows,
            n_cols,
            row_ptr,
            col_idx,
            values,
        }
    }
    
    /// Creates an identity matrix of the given size
    pub fn identity(n: usize) -> Self {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        
        for i in 0..=n {
            row_ptr.push(i);
        }
        
        for i in 0..n {
            col_idx.push(i);
            values.push(T::one());
        }
        
        Self {
            n_rows: n,
            n_cols: n,
            row_ptr,
            col_idx,
            values,
        }
    }
}

impl<T: fmt::Debug + Copy + Num> fmt::Debug for SparseMatrixCSR<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SparseMatrixCSR {{")?;
        writeln!(f, "  dimensions: {} × {}", self.n_rows, self.n_cols)?;
        writeln!(f, "  nnz: {}", self.nnz())?;
        
        // Print a sample of the matrix content
        let max_rows_to_print = 5.min(self.n_rows);
        
        if max_rows_to_print > 0 {
            writeln!(f, "  content sample:")?;
            
            for i in 0..max_rows_to_print {
                write!(f, "    row {}: ", i)?;
                let start = self.row_ptr[i];
                let end = self.row_ptr[i + 1];
                
                if start == end {
                    writeln!(f, "(empty)")?;
                } else {
                    let max_elements = 5.min(end - start);
                    
                    for j in start..(start + max_elements) {
                        write!(f, "({}, {:?}) ", self.col_idx[j], self.values[j])?;
                    }
                    
                    if end - start > max_elements {
                        write!(f, "... ({} more)", end - start - max_elements)?;
                    }
                    
                    writeln!(f)?;
                }
            }
            
            if self.n_rows > max_rows_to_print {
                writeln!(f, "    ... ({} more rows)", self.n_rows - max_rows_to_print)?;
            }
        }
        
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_matrix() {
        let matrix = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 0, 2],
            vec![1, 2, 3, 4, 5],
        );
        
        assert_eq!(matrix.n_rows, 3);
        assert_eq!(matrix.n_cols, 3);
        assert_eq!(matrix.nnz(), 5);
    }
    
    #[test]
    fn test_row_iter() {
        let matrix = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 0, 2],
            vec![1, 2, 3, 4, 5],
        );
        
        let row0: Vec<_> = matrix.row_iter(0).collect();
        assert_eq!(row0, vec![(0, &1), (1, &2)]);
        
        let row1: Vec<_> = matrix.row_iter(1).collect();
        assert_eq!(row1, vec![(1, &3)]);
        
        let row2: Vec<_> = matrix.row_iter(2).collect();
        assert_eq!(row2, vec![(0, &4), (2, &5)]);
    }
    
    #[test]
    fn test_identity() {
        let identity = SparseMatrixCSR::<i32>::identity(3);
        
        assert_eq!(identity.n_rows, 3);
        assert_eq!(identity.n_cols, 3);
        assert_eq!(identity.nnz(), 3);
        
        assert_eq!(identity.row_ptr, vec![0, 1, 2, 3]);
        assert_eq!(identity.col_idx, vec![0, 1, 2]);
        assert_eq!(identity.values, vec![1, 1, 1]);
    }
    
    #[test]
    #[should_panic(expected = "row_ptr.len() must be n_rows + 1")]
    fn test_invalid_row_ptr() {
        SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 3], // Missing last element
            vec![0, 1, 1, 0, 2],
            vec![1, 2, 3, 4, 5],
        );
    }
    
    #[test]
    #[should_panic(expected = "col_idx.len() must equal values.len()")]
    fn test_inconsistent_lengths() {
        SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 0, 2],
            vec![1, 2, 3, 4], // Missing last element
        );
    }
}