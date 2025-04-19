use magnus::{
    analyze_categorization, magnus_spgemm, reference_spgemm, MagnusConfig, SparseMatrixCSR,
};

/// Test that MAGNUS SpGEMM produces the same results as the reference implementation
#[test]
fn test_magnus_spgemm_correctness() {
    // Create two simple sparse matrices
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

    // Create configuration
    let config = MagnusConfig::default();

    // Compute with the reference implementation
    let c_ref = reference_spgemm(&a, &b);

    // Compute with MAGNUS
    let c_magnus = magnus_spgemm(&a, &b, &config);

    // Compare results
    assert_eq!(c_magnus.n_rows, c_ref.n_rows);
    assert_eq!(c_magnus.n_cols, c_ref.n_cols);
    assert_eq!(c_magnus.row_ptr, c_ref.row_ptr);
    assert_eq!(c_magnus.col_idx, c_ref.col_idx);

    // Compare values with a small epsilon for floating point comparisons
    assert_eq!(c_magnus.values.len(), c_ref.values.len());
    for i in 0..c_magnus.values.len() {
        let diff = (c_magnus.values[i] - c_ref.values[i]).abs();
        assert!(diff < 1e-10);
    }
}

/// Test MAGNUS SpGEMM with larger matrices that will use different strategies
#[test]
fn test_magnus_spgemm_strategies() {
    // Create test matrices of different sizes to trigger different strategies
    let a_large = create_large_test_matrix();
    let b_large = create_large_test_matrix();

    // Create configuration
    let config = MagnusConfig::default();

    // Analyze the categorization to confirm we're testing multiple strategies
    let summary = analyze_categorization(&a_large, &b_large, &config);

    // Ensure we have a mix of strategies
    // Note that exact counts will depend on matrix structure and config parameters
    assert!(summary.total_rows > 0);

    // Compute with the reference implementation
    let c_ref = reference_spgemm(&a_large, &b_large);

    // Compute with MAGNUS
    let c_magnus = magnus_spgemm(&a_large, &b_large, &config);

    // Compare results
    assert_eq!(c_magnus.n_rows, c_ref.n_rows);
    assert_eq!(c_magnus.n_cols, c_ref.n_cols);
    assert_eq!(c_magnus.row_ptr, c_ref.row_ptr);

    // The ordering of columns within rows might be different between implementations,
    // so we'll need to do a more sophisticated comparison
    // For this test, we'll just check that the number of non-zeros matches
    for i in 0..c_magnus.n_rows {
        let magnus_row_start = c_magnus.row_ptr[i];
        let magnus_row_end = c_magnus.row_ptr[i + 1];

        let ref_row_start = c_ref.row_ptr[i];
        let ref_row_end = c_ref.row_ptr[i + 1];

        // Check that the row has the same number of non-zeros
        assert_eq!(
            magnus_row_end - magnus_row_start,
            ref_row_end - ref_row_start
        );

        // For each non-zero in the MAGNUS result, check that there's a matching non-zero in the reference
        for j in magnus_row_start..magnus_row_end {
            let col = c_magnus.col_idx[j];
            let val = c_magnus.values[j];

            // Find matching column in reference result
            let mut found = false;
            for k in ref_row_start..ref_row_end {
                if c_ref.col_idx[k] == col {
                    // Check value
                    let diff = (val - c_ref.values[k]).abs();
                    assert!(diff < 1e-10);
                    found = true;
                    break;
                }
            }

            assert!(
                found,
                "Column {} not found in reference result for row {}",
                col, i
            );
        }
    }
}

/// Test MAGNUS with sparse diagonal matrices
#[test]
fn test_magnus_diagonal() {
    // Create diagonal matrices
    let n = 100;
    let a = create_diagonal_matrix(n, 2.0);
    let b = create_diagonal_matrix(n, 3.0);

    // Compute expected result (diagonal matrix with values 6.0)
    let expected_values = vec![6.0; n];

    // Create configuration
    let config = MagnusConfig::default();

    // Compute with MAGNUS
    let c = magnus_spgemm(&a, &b, &config);

    // Check result
    assert_eq!(c.n_rows, n);
    assert_eq!(c.n_cols, n);
    assert_eq!(c.row_ptr, (0..=n).collect::<Vec<usize>>());
    assert_eq!(c.col_idx, (0..n).collect::<Vec<usize>>());

    for i in 0..n {
        let diff = (c.values[i] - expected_values[i]).abs();
        assert!(diff < 1e-10);
    }
}

/// Test MAGNUS with empty matrices
#[test]
fn test_magnus_empty() {
    // Create empty matrices
    let a = SparseMatrixCSR::<f64>::new(5, 5, vec![0, 0, 0, 0, 0, 0], vec![], vec![]);

    let b = SparseMatrixCSR::<f64>::new(5, 5, vec![0, 0, 0, 0, 0, 0], vec![], vec![]);

    // Create configuration
    let config = MagnusConfig::default();

    // Compute with MAGNUS
    let c = magnus_spgemm(&a, &b, &config);

    // Check result
    assert_eq!(c.n_rows, 5);
    assert_eq!(c.n_cols, 5);
    assert_eq!(c.row_ptr, vec![0, 0, 0, 0, 0, 0]);
    assert_eq!(c.col_idx, vec![]);
    assert_eq!(c.values, vec![]);
}

/// Create a diagonal matrix
fn create_diagonal_matrix(n: usize, value: f64) -> SparseMatrixCSR<f64> {
    let row_ptr: Vec<usize> = (0..=n).collect();
    let col_idx: Vec<usize> = (0..n).collect();
    let values = vec![value; n];

    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

/// Create a large test matrix with different row patterns to trigger different strategies
fn create_large_test_matrix() -> SparseMatrixCSR<f64> {
    let n = 1000;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0);
    let mut nnz = 0;

    for i in 0..n {
        let row_nnz = match i % 4 {
            0 => 1,      // Very sparse (1 element) - Sort strategy
            1 => 10,     // Sparse (10 elements) - Sort strategy
            2 => n / 10, // Medium density - Dense or Fine strategy
            3 => n / 2,  // High density - Dense, Fine, or Coarse strategy
            _ => unreachable!(),
        };

        for j in 0..row_nnz {
            // Create a pattern that ensures some rows have very different
            // densities to trigger different strategies
            let col = (i + j * 13) % n;
            col_idx.push(col);
            values.push(1.0 + (i % 10) as f64 / 10.0);
            nnz += 1;
        }

        row_ptr.push(nnz);
    }

    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}
