//! Compressed Sparse Column (CSC) matrix format implementation

use std::fmt;
use num_traits::Num;

/// A sparse matrix in Compressed Sparse Column (CSC) format
///
/// The CSC format stores a sparse matrix using three arrays:
/// - col_ptr: Array of size n_cols + 1 containing indices into row_idx and values arrays
/// - row_idx: Array of size nnz containing row indices of non-zero elements
/// - values: Array of size nnz containing the non-zero values
///
/// This implementation is designed for the MAGNUS sparse matrix multiplication algorithm.
#[derive(Clone)]
pub struct SparseMatrixCSC<T> {
    /// Number of rows in the matrix
    pub n_rows: usize,
    
    /// Number of columns in the matrix
    pub n_cols: usize,
    
    /// Column pointers (size: n_cols + 1)
    /// col_ptr[j] is the index in row_idx and values where column j starts
    /// col_ptr[n_cols] is equal to nnz
    pub col_ptr: Vec<usize>,
    
    /// Row indices (size: nnz)
    pub row_idx: Vec<usize>,
    
    /// Non-zero values (size: nnz)
    pub values: Vec<T>,
}

impl<T> SparseMatrixCSC<T>
where
    T: Copy + Num,
{
    /// Creates a new CSC matrix with the given dimensions and data
    ///
    /// # Arguments
    ///
    /// * `n_rows` - Number of rows
    /// * `n_cols` - Number of columns
    /// * `col_ptr` - Column pointers
    /// * `row_idx` - Row indices
    /// * `values` - Non-zero values
    ///
    /// # Panics
    ///
    /// Panics if the input arrays are inconsistent:
    /// - col_ptr.len() must be n_cols + 1
    /// - row_idx.len() must equal values.len()
    /// - col_ptr[n_cols] must equal row_idx.len()
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        assert_eq!(col_ptr.len(), n_cols + 1, "col_ptr.len() must be n_cols + 1");
        assert_eq!(row_idx.len(), values.len(), "row_idx.len() must equal values.len()");
        assert_eq!(
            col_ptr[n_cols], row_idx.len(),
            "col_ptr[n_cols] must equal row_idx.len()"
        );
        
        // Check that row indices are within bounds
        for &row in &row_idx {
            assert!(row < n_rows, "Row index {} out of bounds (n_rows = {})", row, n_rows);
        }
        
        Self {
            n_rows,
            n_cols,
            col_ptr,
            row_idx,
            values,
        }
    }
    
    /// Returns the number of non-zero elements in the matrix
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Returns an iterator over the non-zero elements in column j
    ///
    /// Each item is a tuple (row_idx, value) representing a non-zero element
    pub fn col_iter(&self, j: usize) -> impl Iterator<Item = (usize, &T)> {
        assert!(j < self.n_cols, "Column index out of bounds");
        
        let start = self.col_ptr[j];
        let end = self.col_ptr[j + 1];
        
        self.row_idx[start..end]
            .iter()
            .zip(&self.values[start..end])
            .map(|(&row, val)| (row, val))
    }
    
    /// Creates an empty matrix with the given dimensions
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        let col_ptr = vec![0; n_cols + 1];
        let row_idx = Vec::new();
        let values = Vec::new();
        
        Self {
            n_rows,
            n_cols,
            col_ptr,
            row_idx,
            values,
        }
    }
    
    /// Creates an identity matrix of the given size
    pub fn identity(n: usize) -> Self {
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        
        for i in 0..=n {
            col_ptr.push(i);
        }
        
        for i in 0..n {
            row_idx.push(i);
            values.push(T::one());
        }
        
        Self {
            n_rows: n,
            n_cols: n,
            col_ptr,
            row_idx,
            values,
        }
    }
}

impl<T: fmt::Debug + Copy + Num> fmt::Debug for SparseMatrixCSC<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SparseMatrixCSC {{")?;
        writeln!(f, "  dimensions: {} × {}", self.n_rows, self.n_cols)?;
        writeln!(f, "  nnz: {}", self.nnz())?;
        
        // Print a sample of the matrix content
        let max_cols_to_print = 5.min(self.n_cols);
        
        if max_cols_to_print > 0 {
            writeln!(f, "  content sample:")?;
            
            for j in 0..max_cols_to_print {
                write!(f, "    col {}: ", j)?;
                let start = self.col_ptr[j];
                let end = self.col_ptr[j + 1];
                
                if start == end {
                    writeln!(f, "(empty)")?;
                } else {
                    let max_elements = 5.min(end - start);
                    
                    for i in start..(start + max_elements) {
                        write!(f, "({}, {:?}) ", self.row_idx[i], self.values[i])?;
                    }
                    
                    if end - start > max_elements {
                        write!(f, "... ({} more)", end - start - max_elements)?;
                    }
                    
                    writeln!(f)?;
                }
            }
            
            if self.n_cols > max_cols_to_print {
                writeln!(f, "    ... ({} more columns)", self.n_cols - max_cols_to_print)?;
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
        let matrix = SparseMatrixCSC::new(
            3, 3,
            vec![0, 2, 4, 5],
            vec![0, 2, 0, 1, 2],
            vec![1, 4, 2, 3, 5],
        );
        
        assert_eq!(matrix.n_rows, 3);
        assert_eq!(matrix.n_cols, 3);
        assert_eq!(matrix.nnz(), 5);
    }
    
    #[test]
    fn test_col_iter() {
        let matrix = SparseMatrixCSC::new(
            3, 3,
            vec![0, 2, 4, 5],
            vec![0, 2, 0, 1, 2],
            vec![1, 4, 2, 3, 5],
        );
        
        let col0: Vec<_> = matrix.col_iter(0).collect();
        assert_eq!(col0, vec![(0, &1), (2, &4)]);
        
        let col1: Vec<_> = matrix.col_iter(1).collect();
        assert_eq!(col1, vec![(0, &2), (1, &3)]);
        
        let col2: Vec<_> = matrix.col_iter(2).collect();
        assert_eq!(col2, vec![(2, &5)]);
    }
    
    #[test]
    fn test_identity() {
        let identity = SparseMatrixCSC::<i32>::identity(3);
        
        assert_eq!(identity.n_rows, 3);
        assert_eq!(identity.n_cols, 3);
        assert_eq!(identity.nnz(), 3);
        
        assert_eq!(identity.col_ptr, vec![0, 1, 2, 3]);
        assert_eq!(identity.row_idx, vec![0, 1, 2]);
        assert_eq!(identity.values, vec![1, 1, 1]);
    }
    
    #[test]
    #[should_panic(expected = "col_ptr.len() must be n_cols + 1")]
    fn test_invalid_col_ptr() {
        SparseMatrixCSC::new(
            3, 3,
            vec![0, 2, 4], // Missing last element
            vec![0, 2, 0, 1, 2],
            vec![1, 4, 2, 3, 5],
        );
    }
    
    #[test]
    #[should_panic(expected = "row_idx.len() must equal values.len()")]
    fn test_inconsistent_lengths() {
        SparseMatrixCSC::new(
            3, 3,
            vec![0, 2, 4, 5],
            vec![0, 2, 0, 1, 2],
            vec![1, 4, 2, 3], // Missing last element
        );
    }
}