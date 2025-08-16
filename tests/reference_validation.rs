//! Validate our reference implementation against sprs (standard Rust library)

use magnus::{reference_spgemm, SparseMatrixCSR};
use sprs::CsMat;

/// Create a simple test matrix
fn create_test_matrix() -> SparseMatrixCSR<f64> {
    // Create a 4x4 matrix:
    // [1 2 0 0]
    // [0 3 4 0]
    // [0 0 5 6]
    // [7 0 0 8]
    let row_ptr = vec![0, 2, 4, 6, 8];
    let col_idx = vec![0, 1, 1, 2, 2, 3, 0, 3];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    SparseMatrixCSR::new(4, 4, row_ptr, col_idx, values)
}

/// Convert to sprs format
fn to_sprs(matrix: &SparseMatrixCSR<f64>) -> CsMat<f64> {
    CsMat::new(
        (matrix.n_rows, matrix.n_cols),
        matrix.row_ptr.clone(),
        matrix.col_idx.clone(),
        matrix.values.clone(),
    )
}

#[test]
fn test_reference_vs_sprs() {
    let a = create_test_matrix();
    let b = create_test_matrix();

    // Compute with our reference implementation
    let result_ref = reference_spgemm(&a, &b);

    // Compute with sprs
    let a_sprs = to_sprs(&a);
    let b_sprs = to_sprs(&b);
    let result_sprs = &a_sprs * &b_sprs;

    // Compare dimensions
    assert_eq!(result_ref.n_rows, result_sprs.rows());
    assert_eq!(result_ref.n_cols, result_sprs.cols());

    // Compare structure
    let sprs_row_ptr: Vec<usize> = result_sprs.indptr().as_slice().unwrap().to_vec();
    let sprs_col_idx: Vec<usize> = result_sprs.indices().to_vec();
    let sprs_values: Vec<f64> = result_sprs.data().to_vec();

    println!("Reference row_ptr: {:?}", result_ref.row_ptr);
    println!("sprs row_ptr: {:?}", sprs_row_ptr);

    assert_eq!(result_ref.row_ptr, sprs_row_ptr, "row_ptr mismatch");
    assert_eq!(result_ref.col_idx, sprs_col_idx, "col_idx mismatch");

    // Compare values (allowing small numerical differences)
    for (ref_val, sprs_val) in result_ref.values.iter().zip(sprs_values.iter()) {
        assert!(
            (ref_val - sprs_val).abs() < 1e-10,
            "Value mismatch: {} vs {}",
            ref_val,
            sprs_val
        );
    }
}

#[test]
fn test_magnus_vs_sprs() {
    use magnus::{magnus_spgemm, MagnusConfig};

    let a = create_test_matrix();
    let b = create_test_matrix();

    // Compute with MAGNUS
    let config = MagnusConfig::default();
    let result_magnus = magnus_spgemm(&a, &b, &config);

    // Compute with sprs
    let a_sprs = to_sprs(&a);
    let b_sprs = to_sprs(&b);
    let result_sprs = &a_sprs * &b_sprs;

    // Compare dimensions
    assert_eq!(result_magnus.n_rows, result_sprs.rows());
    assert_eq!(result_magnus.n_cols, result_sprs.cols());

    // Compare non-zero count
    assert_eq!(result_magnus.nnz(), result_sprs.nnz());

    // Compare values row by row (allows for different ordering within rows)
    for row in 0..result_magnus.n_rows {
        let magnus_start = result_magnus.row_ptr[row];
        let magnus_end = result_magnus.row_ptr[row + 1];

        let sprs_start = result_sprs.indptr().as_slice().unwrap()[row];
        let sprs_end = result_sprs.indptr().as_slice().unwrap()[row + 1];

        // Collect entries for this row
        let mut magnus_entries: Vec<(usize, f64)> = (magnus_start..magnus_end)
            .map(|i| (result_magnus.col_idx[i], result_magnus.values[i]))
            .collect();

        let mut sprs_entries: Vec<(usize, f64)> = (sprs_start..sprs_end)
            .map(|i| (result_sprs.indices()[i], result_sprs.data()[i]))
            .collect();

        // Sort by column for comparison
        magnus_entries.sort_by_key(|&(col, _)| col);
        sprs_entries.sort_by_key(|&(col, _)| col);

        assert_eq!(
            magnus_entries.len(),
            sprs_entries.len(),
            "Row {} has different nnz",
            row
        );

        for ((m_col, m_val), (s_col, s_val)) in magnus_entries.iter().zip(sprs_entries.iter()) {
            assert_eq!(m_col, s_col, "Column mismatch in row {}", row);
            assert!(
                (m_val - s_val).abs() < 1e-10,
                "Value mismatch in row {}, col {}: {} vs {}",
                row,
                m_col,
                m_val,
                s_val
            );
        }
    }
}
