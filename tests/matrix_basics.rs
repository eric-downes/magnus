//! Basic tests for matrix operations and conversions

use magnus::{SparseMatrixCSC, SparseMatrixCSR};

#[test]
fn test_matrix_creation_csr() {
    let matrix = SparseMatrixCSR::new(
        3,
        3,
        vec![0, 2, 3, 5],
        vec![0, 1, 1, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    );

    assert_eq!(matrix.n_rows, 3);
    assert_eq!(matrix.n_cols, 3);
    assert_eq!(matrix.nnz(), 5);

    // Check first row
    let first_row: Vec<_> = matrix.row_iter(0).collect();
    assert_eq!(first_row.len(), 2);
    assert_eq!(first_row[0].0, 0);
    assert_eq!(*first_row[0].1, 1.0);
    assert_eq!(first_row[1].0, 1);
    assert_eq!(*first_row[1].1, 2.0);

    // Check second row
    let second_row: Vec<_> = matrix.row_iter(1).collect();
    assert_eq!(second_row.len(), 1);
    assert_eq!(second_row[0].0, 1);
    assert_eq!(*second_row[0].1, 3.0);

    // Check third row
    let third_row: Vec<_> = matrix.row_iter(2).collect();
    assert_eq!(third_row.len(), 2);
    assert_eq!(third_row[0].0, 0);
    assert_eq!(*third_row[0].1, 4.0);
    assert_eq!(third_row[1].0, 2);
    assert_eq!(*third_row[1].1, 5.0);
}

#[test]
fn test_matrix_creation_csc() {
    let matrix = SparseMatrixCSC::new(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 2, 0, 1, 2],
        vec![1.0, 4.0, 2.0, 3.0, 5.0],
    );

    assert_eq!(matrix.n_rows, 3);
    assert_eq!(matrix.n_cols, 3);
    assert_eq!(matrix.nnz(), 5);

    // Check first column
    let first_col: Vec<_> = matrix.col_iter(0).collect();
    assert_eq!(first_col.len(), 2);
    assert_eq!(first_col[0].0, 0);
    assert_eq!(*first_col[0].1, 1.0);
    assert_eq!(first_col[1].0, 2);
    assert_eq!(*first_col[1].1, 4.0);

    // Check second column
    let second_col: Vec<_> = matrix.col_iter(1).collect();
    assert_eq!(second_col.len(), 2);
    assert_eq!(second_col[0].0, 0);
    assert_eq!(*second_col[0].1, 2.0);
    assert_eq!(second_col[1].0, 1);
    assert_eq!(*second_col[1].1, 3.0);

    // Check third column
    let third_col: Vec<_> = matrix.col_iter(2).collect();
    assert_eq!(third_col.len(), 1);
    assert_eq!(third_col[0].0, 2);
    assert_eq!(*third_col[0].1, 5.0);
}

#[test]
fn test_identity_matrix() {
    let identity = SparseMatrixCSR::<f64>::identity(3);

    assert_eq!(identity.n_rows, 3);
    assert_eq!(identity.n_cols, 3);
    assert_eq!(identity.nnz(), 3);

    // Check that all diagonal elements are 1.0
    for i in 0..3 {
        let row: Vec<_> = identity.row_iter(i).collect();
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].0, i);
        assert_eq!(*row[0].1, 1.0);
    }
}

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
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    );

    // Convert to CSC
    let csc = csr.to_csc();

    // Verify dimensions and nnz
    assert_eq!(csc.n_rows, csr.n_rows);
    assert_eq!(csc.n_cols, csr.n_cols);
    assert_eq!(csc.nnz(), csr.nnz());

    // Check contents of first column
    let col0: Vec<_> = csc.col_iter(0).collect();
    assert_eq!(col0.len(), 2);
    let rows0: Vec<_> = col0.iter().map(|&(row, _)| row).collect();
    assert!(rows0.contains(&0));
    assert!(rows0.contains(&2));

    // Check contents of second column
    let col1: Vec<_> = csc.col_iter(1).collect();
    assert_eq!(col1.len(), 2);
    let rows1: Vec<_> = col1.iter().map(|&(row, _)| row).collect();
    assert!(rows1.contains(&0));
    assert!(rows1.contains(&1));

    // Check contents of third column
    let col2: Vec<_> = csc.col_iter(2).collect();
    assert_eq!(col2.len(), 1);
    assert_eq!(col2[0].0, 2);
}
