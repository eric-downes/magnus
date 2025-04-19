//! Integration tests for accumulators

use magnus::{SparseMatrixCSR, multiply_row_dense, multiply_row_sort, create_accumulator, Accumulator};
use magnus::accumulator::dense::DenseAccumulator;
use magnus::accumulator::sort::SortAccumulator;

#[test]
fn test_dense_accumulator_with_sparse_matrices() {
    // Create test matrices:
    // A = [1 2 0; 0 3 4; 5 0 6]
    // B = [7 0 0; 0 8 0; 0 0 9]
    // Expected result:
    // C = [7 16 0; 0 24 36; 35 0 54]
    
    let a = SparseMatrixCSR::new(
        3, 3,
        vec![0, 2, 4, 6],
        vec![0, 1, 1, 2, 0, 2],
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    
    let b = SparseMatrixCSR::new(
        3, 3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![7.0f64, 8.0, 9.0],
    );
    
    // Get the row categorization for each row (not needed for this test but just to check)
    // let categories = magnus::categorize_rows(&a, &b, &magnus::MagnusConfig::default());
    
    // Test the accumulator operation row by row
    let (cols_row0, vals_row0) = multiply_row_dense(0, &a, &b);
    let (cols_row1, vals_row1) = multiply_row_dense(1, &a, &b);
    let (cols_row2, vals_row2) = multiply_row_dense(2, &a, &b);
    
    // Verify row 0 has expected results
    assert_eq!(cols_row0, vec![0, 1]);
    let diff1: f64 = (vals_row0[0] - 7.0f64).abs();
    let diff2: f64 = (vals_row0[1] - 16.0f64).abs();
    assert!(diff1 < 1.0e-10);
    assert!(diff2 < 1.0e-10);
    
    // Verify row 1 has expected results
    assert_eq!(cols_row1, vec![1, 2]);
    let diff3: f64 = (vals_row1[0] - 24.0f64).abs();
    let diff4: f64 = (vals_row1[1] - 36.0f64).abs();
    assert!(diff3 < 1.0e-10);
    assert!(diff4 < 1.0e-10);
    
    // Verify row 2 has expected results
    assert_eq!(cols_row2, vec![0, 2]);
    let diff5: f64 = (vals_row2[0] - 35.0f64).abs();
    let diff6: f64 = (vals_row2[1] - 54.0f64).abs();
    assert!(diff5 < 1.0e-10);
    assert!(diff6 < 1.0e-10);
}

#[test]
fn test_accumulator_trait_interfaces() {
    // Create concrete accumulators
    let n_cols = 5;
    let _threshold = 256; // Default from paper
    
    // Test using the direct DenseAccumulator implementation
    let mut dense_acc = DenseAccumulator::new(n_cols);
    
    // Accumulate some values
    dense_acc.accumulate(0, 1.0f64);
    dense_acc.accumulate(2, 3.0f64);
    dense_acc.accumulate(4, 5.0f64);
    dense_acc.accumulate(2, 2.0f64); // Duplicate column should be accumulated
    
    // Extract result directly
    let (cols_dense, vals_dense) = dense_acc.extract_result();
    
    // Verify results
    assert_eq!(cols_dense, vec![0, 2, 4]);
    let diff1: f64 = (vals_dense[0] - 1.0f64).abs();
    let diff2: f64 = (vals_dense[1] - 5.0f64).abs(); // 3.0 + 2.0
    let diff3: f64 = (vals_dense[2] - 5.0f64).abs();
    assert!(diff1 < 1.0e-10);
    assert!(diff2 < 1.0e-10);
    assert!(diff3 < 1.0e-10);
    
    // Test using the SortAccumulator implementation with the same input
    let mut sort_acc = SortAccumulator::new(n_cols);
    
    // Accumulate the same values but in a different order
    sort_acc.accumulate(2, 3.0f64);
    sort_acc.accumulate(0, 1.0f64);
    sort_acc.accumulate(2, 2.0f64);
    sort_acc.accumulate(4, 5.0f64);
    
    // Extract result
    let (cols_sort, vals_sort) = sort_acc.extract_result();
    
    // Results should be identical to the dense accumulator, despite different insertion order
    assert_eq!(cols_sort, cols_dense);
    
    for i in 0..cols_dense.len() {
        let diff: f64 = (vals_sort[i] - vals_dense[i]).abs();
        assert!(diff < 1.0e-10, "Values at index {} differ: {} vs {}", i, vals_sort[i], vals_dense[i]);
    }
    
    // Now test the factory function with thresholds
    {
        // Small matrix should use dense accumulator
        let mut small_acc = create_accumulator::<f64>(n_cols, 10);
        
        // Accumulate some values
        small_acc.accumulate(0, 1.0f64);
        small_acc.accumulate(2, 3.0f64);
        small_acc.reset();
        
        // Large matrix should use sort accumulator
        let large_size = 1000;
        let mut large_acc = create_accumulator::<f64>(large_size, 10);
        
        // Accumulate some values
        large_acc.accumulate(0, 1.0f64);
        large_acc.accumulate(999, 3.0f64);
        large_acc.reset();
    }
    
    // Test that concrete accumulators handle reset properly
    {
        // Test reset with dense accumulator
        let mut dense_acc = DenseAccumulator::new(n_cols);
        dense_acc.accumulate(1, 10.0f64);
        dense_acc.accumulate(3, 30.0f64);
        dense_acc.reset();
        
        // Add new values after reset
        dense_acc.accumulate(0, 5.0f64);
        
        // Extract and verify
        let (cols, vals) = dense_acc.extract_result();
        assert_eq!(cols, vec![0]);
        let diff: f64 = (vals[0] - 5.0f64).abs();
        assert!(diff < 1.0e-10);
    }
    
    {
        // Test reset with sort accumulator
        let mut sort_acc = SortAccumulator::new(n_cols);
        sort_acc.accumulate(1, 10.0f64);
        sort_acc.accumulate(3, 30.0f64);
        sort_acc.reset();
        
        // Add new values after reset
        sort_acc.accumulate(0, 5.0f64);
        
        // Extract and verify
        let (cols, vals) = sort_acc.extract_result();
        assert_eq!(cols, vec![0]);
        let diff: f64 = (vals[0] - 5.0f64).abs();
        assert!(diff < 1.0e-10);
    }
}

#[test]
fn test_accumulator_with_empty_row() {
    // Test accumulating an empty row
    let a = SparseMatrixCSR::new(
        3, 3,
        vec![0, 0, 2, 3], // Row 0 is empty
        vec![1, 2, 2],
        vec![3.0f64, 4.0, 5.0],
    );
    
    let b = SparseMatrixCSR::new(
        3, 3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![7.0f64, 8.0, 9.0],
    );
    
    // Test with dense accumulator
    let (cols_row0, vals_row0) = multiply_row_dense(0, &a, &b);
    
    // Result should be empty
    assert_eq!(cols_row0.len(), 0);
    assert_eq!(vals_row0.len(), 0);
    
    // Test with sort-based accumulator
    let (cols_row0_sort, vals_row0_sort) = multiply_row_sort(0, &a, &b);
    
    // Result should be empty
    assert_eq!(cols_row0_sort.len(), 0);
    assert_eq!(vals_row0_sort.len(), 0);
}

#[test]
fn test_accumulator_with_larger_matrices() {
    // Create larger test matrices
    let size = 10;
    
    // Create diagonal matrices
    let mut a_row_ptr = Vec::with_capacity(size + 1);
    let mut a_col_idx = Vec::with_capacity(size);
    let mut a_values = Vec::with_capacity(size);
    
    a_row_ptr.push(0);
    for i in 0..size {
        a_col_idx.push(i);
        a_values.push(2.0f64);
        a_row_ptr.push(a_col_idx.len());
    }
    
    let a = SparseMatrixCSR::new(
        size,
        size,
        a_row_ptr,
        a_col_idx,
        a_values,
    );
    
    let mut b_row_ptr = Vec::with_capacity(size + 1);
    let mut b_col_idx = Vec::with_capacity(size);
    let mut b_values = Vec::with_capacity(size);
    
    b_row_ptr.push(0);
    for i in 0..size {
        b_col_idx.push(i);
        b_values.push(3.0f64);
        b_row_ptr.push(b_col_idx.len());
    }
    
    let b = SparseMatrixCSR::new(
        size,
        size,
        b_row_ptr,
        b_col_idx,
        b_values,
    );
    
    // Test with dense accumulator - result should be diagonal with 6.0 on diagonal
    for i in 0..size {
        let (cols, vals) = multiply_row_dense(i, &a, &b);
        
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0], i);
        let diff: f64 = (vals[0] - 6.0f64).abs();
        assert!(diff < 1.0e-10); // 2.0 * 3.0
    }
    
    // Test with sort-based accumulator - result should be identical
    for i in 0..size {
        let (cols, vals) = multiply_row_sort(i, &a, &b);
        
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0], i);
        let diff: f64 = (vals[0] - 6.0f64).abs();
        assert!(diff < 1.0e-10); // 2.0 * 3.0
    }
}

#[test]
fn test_sort_accumulator_with_complex_matrix() {
    // Create a matrix with repeated column indices in different rows
    // This will test the sort-based accumulator's ability to merge duplicates
    
    // Matrix A: [2 1 0 0; 1 0 3 0; 0 2 1 1; 4 0 0 2]
    let a = SparseMatrixCSR::new(
        4, 4,
        vec![0, 2, 4, 7, 9],
        vec![0, 1, 0, 2, 1, 2, 3, 0, 3],
        vec![2.0f64, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 4.0, 2.0],
    );
    
    // Matrix B: [1 0 2 3; 0 2 1 0; 3 2 0 1; 0 4 0 2]
    let b = SparseMatrixCSR::new(
        4, 4,
        vec![0, 3, 6, 9, 11],
        vec![0, 2, 3, 1, 2, 3, 0, 1, 3, 1, 3],
        vec![1.0f64, 2.0, 3.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 4.0, 2.0],
    );
    
    // Test dense and sort-based accumulators on each row
    for row in 0..4 {
        let (cols_dense, vals_dense) = multiply_row_dense(row, &a, &b);
        let (cols_sort, vals_sort) = multiply_row_sort(row, &a, &b);
        
        // Results should be identical between implementations
        assert_eq!(cols_dense.len(), cols_sort.len(), "Row {} has different number of non-zeros", row);
        assert_eq!(cols_dense, cols_sort, "Row {} has different column indices", row);
        
        for i in 0..cols_dense.len() {
            let diff: f64 = (vals_dense[i] - vals_sort[i]).abs();
            assert!(diff < 1.0e-10, "Row {}, col {} values differ: {} vs {}", 
                  row, cols_dense[i], vals_dense[i], vals_sort[i]);
        }
        
        // Additional verification for specific rows
        match row {
            0 => {
                // Row 0: [2, 2, 5, 6]
                assert_eq!(cols_dense, vec![0, 1, 2, 3]);
                let expected = [2.0f64, 2.0, 5.0, 6.0];
                for i in 0..4 {
                    let diff: f64 = (vals_dense[i] - expected[i]).abs();
                    assert!(diff < 1.0e-10);
                }
            },
            1 => {
                // Row 1: [10, 6, 2, 6]
                assert_eq!(cols_dense, vec![0, 1, 2, 3]);
                let expected = [10.0f64, 6.0, 2.0, 6.0];
                for i in 0..4 {
                    let diff: f64 = (vals_dense[i] - expected[i]).abs();
                    assert!(diff < 1.0e-10);
                }
            },
            // Add other row tests if needed
            _ => {}
        }
    }
}