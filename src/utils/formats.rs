//! Utilities for converting between our matrix formats and external libraries

use crate::matrix::SparseMatrixCSR;
use crate::matrix::SparseMatrixCSC;
use sprs::CsMat;
use num_traits::Num;

/// Converts our CSR matrix format to sprs CsMat format
pub fn to_sprs_csr<T>(matrix: &SparseMatrixCSR<T>) -> CsMat<T>
where 
    T: Copy + Num + Default,
{
    CsMat::new(
        (matrix.n_rows, matrix.n_cols),
        matrix.row_ptr.clone(),
        matrix.col_idx.clone(),
        matrix.values.clone(),
    )
}

/// Converts our CSC matrix format to sprs CsMat format (as CSC)
pub fn to_sprs_csc<T>(matrix: &SparseMatrixCSC<T>) -> CsMat<T>
where 
    T: Copy + Num + Default,
{
    CsMat::new_csc(
        (matrix.n_rows, matrix.n_cols),
        matrix.col_ptr.clone(),
        matrix.row_idx.clone(),
        matrix.values.clone(),
    )
}

/// Converts sprs CsMat in CSR format to our SparseMatrixCSR format
pub fn from_sprs_csr<T>(matrix: CsMat<T>) -> SparseMatrixCSR<T>
where
    T: Copy + Num + Default,
{
    // Ensure matrix is in CSR format
    let matrix = if matrix.is_csr() {
        matrix
    } else {
        matrix.to_csr()
    };
    
    let shape = matrix.shape();
    let (indptr, indices, data) = matrix.into_raw_storage();
    
    SparseMatrixCSR::new(
        shape.0,
        shape.1,
        indptr,
        indices,
        data,
    )
}

/// Converts sprs CsMat in CSC format to our SparseMatrixCSC format
pub fn from_sprs_csc<T>(matrix: CsMat<T>) -> SparseMatrixCSC<T>
where
    T: Copy + Num + Default,
{
    // Ensure matrix is in CSC format
    let matrix = if matrix.is_csc() {
        matrix
    } else {
        matrix.to_csc()
    };
    
    let shape = matrix.shape();
    let (indptr, indices, data) = matrix.into_raw_storage();
    
    SparseMatrixCSC::new(
        shape.0,
        shape.1,
        indptr,
        indices,
        data,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_csr_roundtrip() {
        // Create a test matrix
        let original = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 0, 2],
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
        );
        
        // Convert to sprs and back
        let sprs_mat = to_sprs_csr(&original);
        let roundtrip = from_sprs_csr(sprs_mat);
        
        // Verify dimensions and structure
        assert_eq!(roundtrip.n_rows, original.n_rows);
        assert_eq!(roundtrip.n_cols, original.n_cols);
        assert_eq!(roundtrip.nnz(), original.nnz());
        assert_eq!(roundtrip.row_ptr, original.row_ptr);
        
        // The column indices and values should match after conversion
        for i in 0..original.n_rows {
            let mut original_row: Vec<_> = original.row_iter(i)
                .map(|(col, &val)| (col, val))
                .collect();
            
            let mut roundtrip_row: Vec<_> = roundtrip.row_iter(i)
                .map(|(col, &val)| (col, val))
                .collect();
            
            // Sort by column index for comparison
            original_row.sort_by_key(|&(col, _)| col);
            roundtrip_row.sort_by_key(|&(col, _)| col);
            
            assert_eq!(original_row, roundtrip_row);
        }
    }
    
    #[test]
    fn test_csc_roundtrip() {
        // Create a test matrix
        let original = SparseMatrixCSC::new(
            3, 3,
            vec![0, 2, 4, 5],
            vec![0, 2, 0, 1, 2],
            vec![1.0f64, 4.0, 2.0, 3.0, 5.0],
        );
        
        // Convert to sprs and back
        let sprs_mat = to_sprs_csc(&original);
        let roundtrip = from_sprs_csc(sprs_mat);
        
        // Verify dimensions and structure
        assert_eq!(roundtrip.n_rows, original.n_rows);
        assert_eq!(roundtrip.n_cols, original.n_cols);
        assert_eq!(roundtrip.nnz(), original.nnz());
        assert_eq!(roundtrip.col_ptr, original.col_ptr);
        
        // The row indices and values should match after conversion
        for j in 0..original.n_cols {
            let mut original_col: Vec<_> = original.col_iter(j)
                .map(|(row, &val)| (row, val))
                .collect();
            
            let mut roundtrip_col: Vec<_> = roundtrip.col_iter(j)
                .map(|(row, &val)| (row, val))
                .collect();
            
            // Sort by row index for comparison
            original_col.sort_by_key(|&(row, _)| row);
            roundtrip_col.sort_by_key(|&(row, _)| row);
            
            assert_eq!(original_col, roundtrip_col);
        }
    }
    
    #[test]
    fn test_csr_to_csc_via_sprs() {
        // Create a CSR matrix
        let csr = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 0, 2],
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
        );
        
        // Convert to sprs
        let sprs_mat = to_sprs_csr(&csr);
        
        // Convert to CSC using sprs
        let sprs_csc = sprs_mat.to_csc();
        
        // Convert back to our format
        let csc = from_sprs_csc(sprs_csc);
        
        // Verify dimensions
        assert_eq!(csc.n_rows, csr.n_rows);
        assert_eq!(csc.n_cols, csr.n_cols);
        assert_eq!(csc.nnz(), csr.nnz());
        
        // Verify content - first build a dense representation of both matrices
        let mut dense_csr = vec![vec![0.0f64; csr.n_cols]; csr.n_rows];
        let mut dense_csc = vec![vec![0.0f64; csc.n_cols]; csc.n_rows];
        
        for i in 0..csr.n_rows {
            for (j, &val) in csr.row_iter(i) {
                dense_csr[i][j] = val;
            }
        }
        
        for j in 0..csc.n_cols {
            for (i, &val) in csc.col_iter(j) {
                dense_csc[i][j] = val;
            }
        }
        
        // Compare the dense representations
        for i in 0..csr.n_rows {
            for j in 0..csr.n_cols {
                assert!((dense_csr[i][j] - dense_csc[i][j]).abs() < 1.0e-10);
            }
        }
    }
    
    #[test]
    fn test_sprs_multiply_via_conversion() {
        // Create test matrices:
        // A = [1 2; 0 3]
        // B = [4 5; 6 7]
        // Expected result: C = A*B = [16 19; 18 21]
        
        let a = SparseMatrixCSR::new(
            2, 2,
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![1.0f64, 2.0, 3.0],
        );
        
        let b = SparseMatrixCSR::new(
            2, 2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0f64, 5.0, 6.0, 7.0],
        );
        
        // Convert to sprs
        let sprs_a = to_sprs_csr(&a);
        let sprs_b = to_sprs_csr(&b);
        
        // Multiply using sprs
        let sprs_result = &sprs_a * &sprs_b;
        
        // Convert back to our format
        let result = from_sprs_csr(sprs_result.to_owned());
        
        // Verify dimensions
        assert_eq!(result.n_rows, 2);
        assert_eq!(result.n_cols, 2);
        assert_eq!(result.nnz(), 4);
        
        // Convert to dense for easier verification
        let mut dense_result = vec![vec![0.0f64; 2]; 2];
        for i in 0..2 {
            for (j, &val) in result.row_iter(i) {
                dense_result[i][j] = val;
            }
        }
        
        // Check values
        assert!((dense_result[0][0] - 16.0).abs() < 1.0e-10);
        assert!((dense_result[0][1] - 19.0).abs() < 1.0e-10);
        assert!((dense_result[1][0] - 18.0).abs() < 1.0e-10);
        assert!((dense_result[1][1] - 21.0).abs() < 1.0e-10);
    }
}