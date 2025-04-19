//! Integration tests for format conversions with external libraries

use magnus::SparseMatrixCSR;
use magnus::utils::{to_sprs_csr, to_sprs_csc, from_sprs_csr, from_sprs_csc};

/// Creates a test matrix with a specific pattern
fn create_test_matrix_csr() -> SparseMatrixCSR<f64> {
    // Create a 5x5 matrix with a specific pattern:
    // [ 1.0  0.0  2.0  0.0  0.0 ]
    // [ 0.0  3.0  0.0  0.0  4.0 ]
    // [ 0.0  0.0  5.0  0.0  0.0 ]
    // [ 6.0  0.0  0.0  7.0  0.0 ]
    // [ 0.0  0.0  8.0  0.0  9.0 ]
    
    SparseMatrixCSR::new(
        5, 5,
        vec![0, 2, 4, 5, 7, 9],
        vec![0, 2, 1, 4, 2, 0, 3, 2, 4],
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
}

/// Creates a tridiagonal matrix in CSR format
fn create_tridiagonal_csr(n: usize) -> SparseMatrixCSR<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    
    row_ptr.push(0);
    
    for i in 0..n {
        // Lower diagonal
        if i > 0 {
            col_idx.push(i - 1);
            values.push(1.0f64);
        }
        
        // Diagonal
        col_idx.push(i);
        values.push(2.0f64);
        
        // Upper diagonal
        if i < n - 1 {
            col_idx.push(i + 1);
            values.push(1.0f64);
        }
        
        row_ptr.push(col_idx.len());
    }
    
    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

#[test]
fn test_csr_to_sprs_conversion() {
    let magnus_csr = create_test_matrix_csr();
    
    // Convert to sprs format
    let sprs_mat = to_sprs_csr(&magnus_csr);
    
    // Verify dimensions and nnz
    assert_eq!(sprs_mat.rows(), magnus_csr.n_rows);
    assert_eq!(sprs_mat.cols(), magnus_csr.n_cols);
    assert_eq!(sprs_mat.nnz(), magnus_csr.nnz());
    
    // Verify it's in CSR format
    assert!(sprs_mat.is_csr());
    
    // Verify values by checking a few specific elements
    assert_eq!(sprs_mat.get(0, 0), Some(&1.0));
    assert_eq!(sprs_mat.get(0, 2), Some(&2.0));
    assert_eq!(sprs_mat.get(1, 1), Some(&3.0));
    assert_eq!(sprs_mat.get(2, 2), Some(&5.0));
    assert_eq!(sprs_mat.get(3, 3), Some(&7.0));
    assert_eq!(sprs_mat.get(4, 4), Some(&9.0));
    
    // Check that zeros are really zeros
    assert_eq!(sprs_mat.get(0, 1), None);
    assert_eq!(sprs_mat.get(1, 0), None);
    assert_eq!(sprs_mat.get(2, 3), None);
}

#[test]
fn test_sprs_to_csr_conversion() {
    // Create a sprs matrix directly
    let mut trip = sprs::TriMat::new((4, 4));
    trip.add_triplet(0, 0, 1.0);
    trip.add_triplet(0, 2, 2.0);
    trip.add_triplet(1, 1, 3.0);
    trip.add_triplet(2, 0, 4.0);
    trip.add_triplet(3, 3, 5.0);
    
    let sprs_mat = trip.to_csr();
    
    // Convert to our format
    let magnus_csr = from_sprs_csr(sprs_mat);
    
    // Verify dimensions and nnz
    assert_eq!(magnus_csr.n_rows, 4);
    assert_eq!(magnus_csr.n_cols, 4);
    assert_eq!(magnus_csr.nnz(), 5);
    
    // Verify values by building a dense representation
    let mut dense = vec![vec![0.0; 4]; 4];
    for i in 0..4 {
        for (j, &val) in magnus_csr.row_iter(i) {
            dense[i][j] = val;
        }
    }
    
    assert_eq!(dense[0][0], 1.0);
    assert_eq!(dense[0][2], 2.0);
    assert_eq!(dense[1][1], 3.0);
    assert_eq!(dense[2][0], 4.0);
    assert_eq!(dense[3][3], 5.0);
}

#[test]
fn test_csc_conversions() {
    // Start with a CSR matrix
    let csr = create_test_matrix_csr();
    
    // Convert to our CSC format
    let csc = csr.to_csc();
    
    // Convert to sprs CSC format
    let sprs_csc = to_sprs_csc(&csc);
    
    // Verify it's in CSC format
    assert!(sprs_csc.is_csc());
    
    // Convert back to our CSC format
    let roundtrip_csc = from_sprs_csc(sprs_csc);
    
    // Verify dimensions and structure
    assert_eq!(roundtrip_csc.n_rows, csc.n_rows);
    assert_eq!(roundtrip_csc.n_cols, csc.n_cols);
    assert_eq!(roundtrip_csc.nnz(), csc.nnz());
    
    // Verify the column pointers match
    assert_eq!(roundtrip_csc.col_ptr, csc.col_ptr);
    
    // Verify the content
    for j in 0..csc.n_cols {
        let mut original_col: Vec<_> = csc.col_iter(j)
            .map(|(row, &val)| (row, val))
            .collect();
        
        let mut roundtrip_col: Vec<_> = roundtrip_csc.col_iter(j)
            .map(|(row, &val)| (row, val))
            .collect();
        
        // Sort by row index for comparison
        original_col.sort_by_key(|&(row, _)| row);
        roundtrip_col.sort_by_key(|&(row, _)| row);
        
        assert_eq!(original_col, roundtrip_col);
    }
}

#[test]
fn test_sprs_multiplication() {
    // Create a tridiagonal matrix
    let a = create_tridiagonal_csr(10);
    
    // Convert to sprs
    let sprs_a = to_sprs_csr(&a);
    
    // Multiply with itself using sprs
    let sprs_result = &sprs_a * &sprs_a;
    
    // Convert back to our format
    let result = from_sprs_csr(sprs_result.to_owned());
    
    // Verify dimensions
    assert_eq!(result.n_rows, 10);
    assert_eq!(result.n_cols, 10);
    
    // Verify the structure - should now be pentadiagonal
    for i in 0..10 {
        let row: Vec<_> = result.row_iter(i).collect();
        let cols: Vec<_> = row.iter().map(|&(col, _)| col).collect();
        
        // Outer rows have fewer non-zeros
        match i {
            0 => {
                assert_eq!(row.len(), 3); // First row: diagonal + 2 above
                assert!(cols.contains(&0));
                assert!(cols.contains(&1));
                assert!(cols.contains(&2));
            },
            1 => {
                assert_eq!(row.len(), 4); // Second row: 1 below + diagonal + 2 above
                assert!(cols.contains(&0));
                assert!(cols.contains(&1));
                assert!(cols.contains(&2));
                assert!(cols.contains(&3));
            },
            8 => {
                assert_eq!(row.len(), 4); // Second-to-last row: 2 below + diagonal + 1 above
                assert!(cols.contains(&6));
                assert!(cols.contains(&7));
                assert!(cols.contains(&8));
                assert!(cols.contains(&9));
            },
            9 => {
                assert_eq!(row.len(), 3); // Last row: 2 below + diagonal
                assert!(cols.contains(&7));
                assert!(cols.contains(&8));
                assert!(cols.contains(&9));
            },
            i if i >= 2 && i <= 7 => {
                assert_eq!(row.len(), 5); // Middle rows: 2 below + diagonal + 2 above
                assert!(cols.contains(&(i-2)));
                assert!(cols.contains(&(i-1)));
                assert!(cols.contains(&i));
                assert!(cols.contains(&(i+1)));
                assert!(cols.contains(&(i+2)));
            },
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_large_banded_matrix() {
    // Create a large banded matrix (tridiagonal)
    let size = 1000;
    let a = create_tridiagonal_csr(size);
    
    // Verify it has the right structure
    assert_eq!(a.n_rows, size);
    assert_eq!(a.n_cols, size);
    assert_eq!(a.nnz(), 3*size - 2); // Tridiagonal with size elements
    
    // Convert to sprs
    let sprs_a = to_sprs_csr(&a);
    
    // Verify the sprs matrix has the same structure
    assert_eq!(sprs_a.rows(), size);
    assert_eq!(sprs_a.cols(), size);
    assert_eq!(sprs_a.nnz(), 3*size - 2);
    
    // Check specific values
    assert_eq!(sprs_a.get(0, 0), Some(&2.0));
    assert_eq!(sprs_a.get(0, 1), Some(&1.0));
    assert_eq!(sprs_a.get(size/2, size/2-1), Some(&1.0));
    assert_eq!(sprs_a.get(size/2, size/2), Some(&2.0));
    assert_eq!(sprs_a.get(size/2, size/2+1), Some(&1.0));
    assert_eq!(sprs_a.get(size-1, size-2), Some(&1.0));
    assert_eq!(sprs_a.get(size-1, size-1), Some(&2.0));
    
    // Convert back
    let roundtrip = from_sprs_csr(sprs_a);
    
    // Verify the structure is preserved
    assert_eq!(roundtrip.n_rows, size);
    assert_eq!(roundtrip.n_cols, size);
    assert_eq!(roundtrip.nnz(), 3*size - 2);
}