//! Tests for SpGEMM correctness against reference implementation

use magnus::{magnus_spgemm, reference_spgemm, MagnusConfig, SparseMatrixCSR};

/// Create a diagonal matrix
fn create_diagonal_matrix(n: usize, value: f64) -> SparseMatrixCSR<f64> {
    let row_ptr: Vec<usize> = (0..=n).collect();
    let col_idx: Vec<usize> = (0..n).collect();
    let values = vec![value; n];

    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

/// Create a tridiagonal matrix
fn create_tridiagonal_matrix(n: usize) -> SparseMatrixCSR<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0);

    for i in 0..n {
        // Lower diagonal
        if i > 0 {
            col_idx.push(i - 1);
            values.push(1.0);
        }

        // Diagonal
        col_idx.push(i);
        values.push(2.0);

        // Upper diagonal
        if i < n - 1 {
            col_idx.push(i + 1);
            values.push(1.0);
        }

        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

#[test]
fn test_identity_multiplication() {
    let a = create_diagonal_matrix(10, 1.0); // Identity
    let b = create_diagonal_matrix(10, 2.0); // Diagonal with 2.0

    // Identity * B should equal B
    let result = reference_spgemm(&a, &b);

    // Check dimensions
    assert_eq!(result.n_rows, 10);
    assert_eq!(result.n_cols, 10);
    assert_eq!(result.nnz(), 10);

    // Check all diagonal elements are 2.0
    for i in 0..10 {
        let row: Vec<_> = result.row_iter(i).collect();
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].0, i);
        assert_eq!(*row[0].1, 2.0);
    }
}

#[test]
fn test_diagonal_multiplication() {
    let a = create_diagonal_matrix(5, 2.0);
    let b = create_diagonal_matrix(5, 3.0);

    // Result should be a diagonal with 6.0
    let result = reference_spgemm(&a, &b);

    assert_eq!(result.n_rows, 5);
    assert_eq!(result.n_cols, 5);
    assert_eq!(result.nnz(), 5);

    for i in 0..5 {
        let row: Vec<_> = result.row_iter(i).collect();
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].0, i);
        assert_eq!(*row[0].1, 6.0);
    }
}

#[test]
fn test_tridiagonal_multiplication() {
    let a = create_tridiagonal_matrix(5);
    let b = create_tridiagonal_matrix(5);

    // Tridiagonal * Tridiagonal should produce a pentadiagonal matrix
    let result = reference_spgemm(&a, &b);

    assert_eq!(result.n_rows, 5);
    assert_eq!(result.n_cols, 5);

    // Verify the structure of the result
    for i in 0..5 {
        let row: Vec<_> = result.row_iter(i).collect();

        // Number of non-zeros in each row depends on position
        match i {
            0 => {
                assert_eq!(row.len(), 3); // First row: 3 non-zeros
                let cols: Vec<_> = row.iter().map(|&(col, _)| col).collect();
                assert!(cols.contains(&0));
                assert!(cols.contains(&1));
                assert!(cols.contains(&2));
            }
            1 => {
                assert_eq!(row.len(), 4); // Second row: 4 non-zeros
                let cols: Vec<_> = row.iter().map(|&(col, _)| col).collect();
                assert!(cols.contains(&0));
                assert!(cols.contains(&1));
                assert!(cols.contains(&2));
                assert!(cols.contains(&3));
            }
            2 => {
                assert_eq!(row.len(), 5); // Middle row: 5 non-zeros
                let cols: Vec<_> = row.iter().map(|&(col, _)| col).collect();
                assert!(cols.contains(&0));
                assert!(cols.contains(&1));
                assert!(cols.contains(&2));
                assert!(cols.contains(&3));
                assert!(cols.contains(&4));
            }
            3 => {
                assert_eq!(row.len(), 4); // Fourth row: 4 non-zeros
                let cols: Vec<_> = row.iter().map(|&(col, _)| col).collect();
                assert!(cols.contains(&1));
                assert!(cols.contains(&2));
                assert!(cols.contains(&3));
                assert!(cols.contains(&4));
            }
            4 => {
                assert_eq!(row.len(), 3); // Last row: 3 non-zeros
                let cols: Vec<_> = row.iter().map(|&(col, _)| col).collect();
                assert!(cols.contains(&2));
                assert!(cols.contains(&3));
                assert!(cols.contains(&4));
            }
            _ => unreachable!(),
        }
    }
}

// Test that MAGNUS produces the same results as the reference implementation
#[test]
fn test_magnus_against_reference() {
    let a = create_tridiagonal_matrix(10);
    let b = create_diagonal_matrix(10, 2.0);

    let config = MagnusConfig::default();

    // Compute with both implementations
    let result_reference = reference_spgemm(&a, &b);
    let result_magnus = magnus_spgemm(&a, &b, &config);

    // Check dimensions match
    assert_eq!(result_magnus.n_rows, result_reference.n_rows);
    assert_eq!(result_magnus.n_cols, result_reference.n_cols);
    assert_eq!(result_magnus.nnz(), result_reference.nnz());

    // Check structure and values match
    for i in 0..result_reference.n_rows {
        let ref_row: Vec<_> = result_reference
            .row_iter(i)
            .map(|(col, &val)| (col, val))
            .collect();

        let magnus_row: Vec<_> = result_magnus
            .row_iter(i)
            .map(|(col, &val)| (col, val))
            .collect();

        assert_eq!(ref_row.len(), magnus_row.len());

        // Sort by column for comparison
        let mut ref_row = ref_row;
        let mut magnus_row = magnus_row;
        ref_row.sort_by_key(|&(col, _)| col);
        magnus_row.sort_by_key(|&(col, _)| col);

        for j in 0..ref_row.len() {
            assert_eq!(ref_row[j].0, magnus_row[j].0);
            assert!((ref_row[j].1 - magnus_row[j].1).abs() < 1e-10);
        }
    }
}
