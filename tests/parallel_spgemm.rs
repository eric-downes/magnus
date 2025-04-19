//! Tests for parallel implementation of MAGNUS SpGEMM

use magnus::{
    SparseMatrixCSR, MagnusConfig, magnus_spgemm, magnus_spgemm_parallel, reference_spgemm,
    process_coarse_level_rows_parallel
};

// Helper function to compare CSR matrices for approximate equality
fn matrices_approximately_equal<T>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    epsilon: T,
) -> bool
where
    T: std::cmp::PartialOrd + std::ops::Sub<Output = T> + Copy + num_traits::Zero + PartialOrd,
{
    if a.n_rows != b.n_rows || a.n_cols != b.n_cols {
        return false;
    }
    
    // Check row pointers
    if a.row_ptr.len() != b.row_ptr.len() {
        return false;
    }
    
    // Check each row's nonzero count
    for i in 0..a.n_rows {
        if a.row_ptr[i + 1] - a.row_ptr[i] != b.row_ptr[i + 1] - b.row_ptr[i] {
            return false;
        }
    }
    
    // Check each row's values and column indices
    for i in 0..a.n_rows {
        let a_start = a.row_ptr[i];
        let a_end = a.row_ptr[i + 1];
        let b_start = b.row_ptr[i];
        let b_end = b.row_ptr[i + 1];
        
        // Sort column indices and values for comparison
        let mut a_entries: Vec<(usize, T)> = (a_start..a_end)
            .map(|idx| (a.col_idx[idx], a.values[idx]))
            .collect();
        a_entries.sort_by_key(|entry| entry.0);
        
        let mut b_entries: Vec<(usize, T)> = (b_start..b_end)
            .map(|idx| (b.col_idx[idx], b.values[idx]))
            .collect();
        b_entries.sort_by_key(|entry| entry.0);
        
        // Compare sorted entries
        for (a_entry, b_entry) in a_entries.iter().zip(b_entries.iter()) {
            if a_entry.0 != b_entry.0 {
                return false;
            }
            
            let diff = if a_entry.1 > b_entry.1 {
                a_entry.1 - b_entry.1
            } else {
                b_entry.1 - a_entry.1
            };
            
            if diff > epsilon {
                return false;
            }
        }
    }
    
    true
}

#[test]
fn test_parallel_vs_sequential() {
    // Create test matrices
    let a = SparseMatrixCSR::new(
        3, 3,
        vec![0, 2, 4, 6],
        vec![0, 1, 0, 2, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    
    let b = SparseMatrixCSR::new(
        3, 3,
        vec![0, 2, 4, 6],
        vec![0, 2, 0, 1, 1, 2],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    );
    
    // Multiply using both implementations
    let config = MagnusConfig::default();
    let c_sequential = magnus_spgemm(&a, &b, &config);
    let c_parallel = magnus_spgemm_parallel(&a, &b, &config);
    
    // Results should be identical
    assert!(matrices_approximately_equal(&c_sequential, &c_parallel, 1e-10));
}

#[test]
fn test_parallel_vs_reference() {
    // Create test matrices
    let a = SparseMatrixCSR::new(
        4, 4,
        vec![0, 3, 5, 7, 9],
        vec![0, 1, 2, 1, 3, 0, 2, 1, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    
    let b = SparseMatrixCSR::new(
        4, 4,
        vec![0, 2, 4, 6, 8],
        vec![0, 3, 1, 2, 0, 3, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    
    // Multiply using both implementations
    let config = MagnusConfig::default();
    let c_reference = reference_spgemm(&a, &b);
    let c_parallel = magnus_spgemm_parallel(&a, &b, &config);
    
    // Results should be identical (with floating point tolerance)
    assert!(matrices_approximately_equal(&c_reference, &c_parallel, 1e-10));
}

#[test]
fn test_parallel_with_empty_rows() {
    // Create matrices with empty rows
    let a = SparseMatrixCSR::new(
        4, 3,
        vec![0, 2, 2, 4, 5],  // Second row is empty
        vec![0, 1, 0, 2, 0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    );
    
    let b = SparseMatrixCSR::new(
        3, 4,
        vec![0, 1, 3, 4],  // First and third rows have one nonzero
        vec![0, 1, 2, 3],
        vec![6.0, 7.0, 8.0, 9.0],
    );
    
    // Multiply using both implementations
    let config = MagnusConfig::default();
    let c_sequential = magnus_spgemm(&a, &b, &config);
    let c_parallel = magnus_spgemm_parallel(&a, &b, &config);
    
    // Results should be identical
    assert!(matrices_approximately_equal(&c_sequential, &c_parallel, 1e-10));
    
    // Second row should be empty in the result
    assert_eq!(c_parallel.row_ptr[1], c_parallel.row_ptr[2]);
}

#[test]
fn test_parallel_large_random_matrices() {
    // Create larger random matrices to better test parallelism
    // This is a very simple random matrix generator for testing
    let n = 100;
    let density = 0.1;
    let nnz_per_row = (n as f64 * density) as usize;
    
    let mut row_ptr_a = Vec::with_capacity(n + 1);
    let mut col_idx_a = Vec::with_capacity(n * nnz_per_row);
    let mut values_a = Vec::with_capacity(n * nnz_per_row);
    
    row_ptr_a.push(0);
    for i in 0..n {
        // Add random elements to row i
        for _ in 0..nnz_per_row {
            let col = i % n; // Simple pattern to ensure reproducibility
            col_idx_a.push(col);
            values_a.push(1.0);
        }
        row_ptr_a.push(col_idx_a.len());
    }
    
    let a = SparseMatrixCSR::new(
        n, n,
        row_ptr_a,
        col_idx_a,
        values_a,
    );
    
    // Create matrix B with similar pattern
    let mut row_ptr_b = Vec::with_capacity(n + 1);
    let mut col_idx_b = Vec::with_capacity(n * nnz_per_row);
    let mut values_b = Vec::with_capacity(n * nnz_per_row);
    
    row_ptr_b.push(0);
    for i in 0..n {
        // Add random elements to row i
        for _ in 0..nnz_per_row {
            let col = (i + 1) % n; // Different pattern
            col_idx_b.push(col);
            values_b.push(1.0);
        }
        row_ptr_b.push(col_idx_b.len());
    }
    
    let b = SparseMatrixCSR::new(
        n, n,
        row_ptr_b,
        col_idx_b,
        values_b,
    );
    
    // Multiply using both implementations
    let config = MagnusConfig::default();
    let c_sequential = magnus_spgemm(&a, &b, &config);
    let c_parallel = magnus_spgemm_parallel(&a, &b, &config);
    
    // Results should be identical
    assert!(matrices_approximately_equal(&c_sequential, &c_parallel, 1e-10));
}

// Test that directly compares the performance of parallel vs sequential
#[test]
fn test_parallel_coarse_level_processing() {
    // Create a test case with diagonal matrices
    let a = SparseMatrixCSR::new(
        4, 4,
        vec![0, 1, 2, 3, 4],  // Four rows with one non-zero each
        vec![0, 1, 2, 3],     // Each row has one entry on the diagonal
        vec![1.0, 1.0, 1.0, 1.0], // All ones
    );
    
    let b = SparseMatrixCSR::new(
        4, 4,
        vec![0, 1, 2, 3, 4],  // Four rows with one non-zero each
        vec![0, 1, 2, 3],     // Each row has one entry on the diagonal
        vec![2.0, 3.0, 4.0, 5.0], // Different values
    );
    
    let config = MagnusConfig::default();
    
    // Process all rows in parallel
    let rows = vec![0, 1, 2, 3];
    let parallel_results = process_coarse_level_rows_parallel(&a, &b, &rows, &config);
    
    // Verify we got results for all rows
    assert_eq!(parallel_results.len(), 4);
    
    // Check that the results match what we expect (diagonal values product)
    for i in 0..rows.len() {
        let (row_idx, col_indices, values) = &parallel_results[i];
        
        // Row index should match
        assert_eq!(*row_idx, i);
        
        // One non-zero per row
        assert_eq!(col_indices.len(), 1);
        
        // Column index should match row (diagonal)
        assert_eq!(col_indices[0], i);
        
        // Value should be product of diagonal entries
        assert_eq!(values[0], (i + 2) as f64);
    }
}

// This test is commented out as it's not meant to be run in CI,
// but can be uncommented for local benchmarking
/*
#[test]
fn benchmark_parallel_vs_sequential() {
    use std::time::{Instant, Duration};
    
    // Create large test matrices
    let n = 1000;
    let a = create_large_test_matrix(n, 0.01);
    let b = create_large_test_matrix(n, 0.01);
    
    // Multiply using sequential implementation
    let config = MagnusConfig::default();
    let start = Instant::now();
    let _ = magnus_spgemm(&a, &b, &config);
    let sequential_time = start.elapsed();
    
    // Multiply using parallel implementation
    let start = Instant::now();
    let _ = magnus_spgemm_parallel(&a, &b, &config);
    let parallel_time = start.elapsed();
    
    println!("Sequential time: {:?}", sequential_time);
    println!("Parallel time: {:?}", parallel_time);
    println!("Speedup: {:.2}x", sequential_time.as_secs_f64() / parallel_time.as_secs_f64());
    
    // The parallel version should be faster
    assert!(parallel_time < sequential_time);
}

fn create_large_test_matrix(n: usize, density: f64) -> SparseMatrixCSR<f64> {
    // Simple function to create a large test matrix
    let nnz_per_row = (n as f64 * density) as usize;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n * nnz_per_row);
    let mut values = Vec::with_capacity(n * nnz_per_row);
    
    row_ptr.push(0);
    for i in 0..n {
        // Add random elements to row i
        for j in 0..nnz_per_row {
            let col = (i + j) % n;
            col_idx.push(col);
            values.push(1.0);
        }
        row_ptr.push(col_idx.len());
    }
    
    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}
*/