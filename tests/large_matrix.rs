/// Tests for large sparse matrices (>1M non-zeros)
/// 
/// These tests validate MAGNUS correctness and performance characteristics
/// on matrices that represent real-world use cases.

use magnus::{magnus_spgemm, magnus_spgemm_parallel, MagnusConfig, SparseMatrixCSR};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Generate a large sparse matrix with specified dimensions and density
fn generate_large_sparse_matrix(
    rows: usize,
    cols: usize,
    density: f64,
    seed: u64,
) -> SparseMatrixCSR<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let nnz_target = ((rows * cols) as f64 * density) as usize;
    
    // Build in CSR format directly for efficiency
    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx = Vec::with_capacity(nnz_target);
    let mut values = Vec::with_capacity(nnz_target);
    
    row_ptr.push(0);
    
    for _ in 0..rows {
        // Determine number of non-zeros for this row (Poisson-like distribution)
        let row_nnz = rng.gen_range(0..=(2.0 * nnz_target as f64 / rows as f64) as usize);
        let row_nnz = row_nnz.min(cols); // Can't have more than cols non-zeros
        
        // Generate sorted column indices for this row
        let mut row_cols = Vec::with_capacity(row_nnz);
        let mut used = std::collections::HashSet::new();
        
        for _ in 0..row_nnz {
            let mut col = rng.gen_range(0..cols);
            while used.contains(&col) {
                col = rng.gen_range(0..cols);
            }
            used.insert(col);
            row_cols.push(col);
        }
        
        row_cols.sort_unstable();
        
        // Add to matrix
        for col in row_cols {
            col_idx.push(col);
            values.push(rng.gen_range(-10.0..10.0));
        }
        
        row_ptr.push(col_idx.len());
    }
    
    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

/// Generate a power-law distributed sparse matrix (common in graph applications)
fn generate_power_law_matrix(
    n: usize,
    avg_degree: usize,
    seed: u64,
) -> SparseMatrixCSR<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    
    // Generate degree sequence following power law
    let mut degrees = Vec::with_capacity(n);
    let total_edges = n * avg_degree;
    
    // Simple power law: P(degree = k) ~ k^(-2.5)
    for i in 0..n {
        let base_prob = 1.0 / ((i + 1) as f64).powf(0.5);
        let degree = (avg_degree as f64 * base_prob * rng.gen_range(0.5..2.0)) as usize;
        degrees.push(degree.min(n / 10)); // Cap at n/10 to avoid extremely dense rows
    }
    
    // Normalize to get exactly the target number of edges
    let sum: usize = degrees.iter().sum();
    if sum > 0 {
        for i in 0..n {
            degrees[i] = (degrees[i] * total_edges / sum).max(1);
        }
    }
    
    // Build the matrix
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(total_edges);
    let mut values = Vec::with_capacity(total_edges);
    
    row_ptr.push(0);
    
    for i in 0..n {
        let degree = degrees[i].min(n);
        let mut row_cols = Vec::with_capacity(degree);
        let mut used = std::collections::HashSet::new();
        
        for _ in 0..degree {
            let mut col = rng.gen_range(0..n);
            let mut attempts = 0;
            while used.contains(&col) && attempts < 100 {
                col = rng.gen_range(0..n);
                attempts += 1;
            }
            if attempts < 100 {
                used.insert(col);
                row_cols.push(col);
            }
        }
        
        row_cols.sort_unstable();
        
        for col in row_cols {
            col_idx.push(col);
            values.push(rng.gen_range(0.1..10.0));
        }
        
        row_ptr.push(col_idx.len());
    }
    
    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

/// Test correctness on a moderately large matrix (>1M nnz)
#[test]
fn test_large_matrix_correctness_1m() {
    // Create a 10,000 x 10,000 matrix with ~1% density = ~1M non-zeros
    let a = generate_large_sparse_matrix(10_000, 10_000, 0.01, 42);
    let b = generate_large_sparse_matrix(10_000, 10_000, 0.01, 43);
    
    let nnz_a = a.col_idx.len();
    let nnz_b = b.col_idx.len();
    println!("Matrix A: {} x {} with {} non-zeros", a.n_rows, a.n_cols, nnz_a);
    println!("Matrix B: {} x {} with {} non-zeros", b.n_rows, b.n_cols, nnz_b);
    
    let config = MagnusConfig::default();
    
    // Compute using MAGNUS
    let c = magnus_spgemm(&a, &b, &config);
    
    println!("Result C: {} x {} with {} non-zeros", c.n_rows, c.n_cols, c.col_idx.len());
    
    // Basic sanity checks
    assert_eq!(c.n_rows, a.n_rows);
    assert_eq!(c.n_cols, b.n_cols);
    assert_eq!(c.row_ptr.len(), c.n_rows + 1);
    assert_eq!(c.row_ptr[0], 0);
    assert_eq!(c.row_ptr[c.n_rows], c.col_idx.len());
    
    // Check that columns are sorted within each row
    for i in 0..c.n_rows {
        let start = c.row_ptr[i];
        let end = c.row_ptr[i + 1];
        if end > start {
            for j in start..end-1 {
                assert!(c.col_idx[j] < c.col_idx[j + 1], 
                    "Columns not sorted in row {}", i);
            }
        }
    }
}

/// Test parallel execution on large matrices
#[test]
fn test_large_matrix_parallel_2m() {
    // Create larger matrices with ~2M non-zeros
    let a = generate_large_sparse_matrix(15_000, 15_000, 0.009, 44);
    let b = generate_large_sparse_matrix(15_000, 15_000, 0.009, 45);
    
    let nnz_a = a.col_idx.len();
    let nnz_b = b.col_idx.len();
    println!("Matrix A: {} x {} with {} non-zeros", a.n_rows, a.n_cols, nnz_a);
    println!("Matrix B: {} x {} with {} non-zeros", b.n_rows, b.n_cols, nnz_b);
    
    let config = MagnusConfig::default();
    
    // Compute using parallel MAGNUS
    let c_parallel = magnus_spgemm_parallel(&a, &b, &config);
    
    // Compute using serial MAGNUS for comparison
    let c_serial = magnus_spgemm(&a, &b, &config);
    
    println!("Parallel result: {} non-zeros", c_parallel.col_idx.len());
    println!("Serial result: {} non-zeros", c_serial.col_idx.len());
    
    // Results should be identical
    assert_eq!(c_parallel.n_rows, c_serial.n_rows);
    assert_eq!(c_parallel.n_cols, c_serial.n_cols);
    assert_eq!(c_parallel.row_ptr, c_serial.row_ptr);
    
    // Note: Column ordering within rows might differ due to parallelism,
    // so we need a more sophisticated comparison
    for i in 0..c_parallel.n_rows {
        let start_p = c_parallel.row_ptr[i];
        let end_p = c_parallel.row_ptr[i + 1];
        let start_s = c_serial.row_ptr[i];
        let end_s = c_serial.row_ptr[i + 1];
        
        assert_eq!(end_p - start_p, end_s - start_s, 
            "Row {} has different nnz count", i);
        
        // Create sorted (col, val) pairs for comparison
        let mut row_p: Vec<_> = (start_p..end_p)
            .map(|j| (c_parallel.col_idx[j], c_parallel.values[j]))
            .collect();
        let mut row_s: Vec<_> = (start_s..end_s)
            .map(|j| (c_serial.col_idx[j], c_serial.values[j]))
            .collect();
        
        row_p.sort_by_key(|&(col, _)| col);
        row_s.sort_by_key(|&(col, _)| col);
        
        for (p, s) in row_p.iter().zip(row_s.iter()) {
            assert_eq!(p.0, s.0, "Column mismatch in row {}", i);
            assert!((p.1 - s.1).abs() < 1e-10, 
                "Value mismatch in row {} col {}", i, p.0);
        }
    }
}

/// Test power-law matrices (common in graph applications)
#[test]
fn test_power_law_matrix_5m() {
    // Create a 50,000 node graph with average degree 100 = ~5M edges
    let a = generate_power_law_matrix(50_000, 100, 46);
    let b = generate_power_law_matrix(50_000, 100, 47);
    
    let nnz_a = a.col_idx.len();
    let nnz_b = b.col_idx.len();
    println!("Power-law A: {} x {} with {} non-zeros", a.n_rows, a.n_cols, nnz_a);
    println!("Power-law B: {} x {} with {} non-zeros", b.n_rows, b.n_cols, nnz_b);
    
    let config = MagnusConfig::default();
    
    // Use parallel version for large matrices
    let c = magnus_spgemm_parallel(&a, &b, &config);
    
    println!("Result: {} non-zeros", c.col_idx.len());
    
    // Validate structure
    assert_eq!(c.n_rows, a.n_rows);
    assert_eq!(c.n_cols, b.n_cols);
    
    // Check for reasonable fill-in (power-law matrices can have significant fill)
    let fill_factor = c.col_idx.len() as f64 / nnz_a.max(nnz_b) as f64;
    println!("Fill factor: {:.2}x", fill_factor);
    assert!(fill_factor > 0.5, "Unexpectedly low fill factor");
    assert!(fill_factor < 100.0, "Unexpectedly high fill factor");
}

/// Test extremely sparse matrix (stress test for sort accumulator)
#[test]
fn test_ultra_sparse_10m() {
    // 100,000 x 100,000 with 0.1% density = 10M non-zeros
    let a = generate_large_sparse_matrix(100_000, 100_000, 0.001, 48);
    let nnz = a.col_idx.len();
    println!("Ultra-sparse matrix: {} x {} with {} non-zeros", 
        a.n_rows, a.n_cols, nnz);
    
    // Square the matrix (A * A)
    let config = MagnusConfig::default();
    let c = magnus_spgemm_parallel(&a, &a, &config);
    
    println!("Result: {} non-zeros", c.col_idx.len());
    
    // Check basic properties
    assert_eq!(c.n_rows, a.n_rows);
    assert_eq!(c.n_cols, a.n_cols);
    
    // For ultra-sparse matrices, the result should still be quite sparse
    let density = c.col_idx.len() as f64 / (c.n_rows * c.n_cols) as f64;
    println!("Result density: {:.6}%", density * 100.0);
    assert!(density < 0.01, "Result too dense for ultra-sparse input");
}

/// Test memory usage patterns with very large matrices
#[test]
#[ignore] // Run with --ignored flag for memory-intensive tests
fn test_memory_scaling_20m() {
    // This test is designed to stress memory management
    let sizes = vec![
        (100_000, 100_000, 0.002), // ~20M nnz
        (150_000, 150_000, 0.001), // ~22.5M nnz  
    ];
    
    for (rows, cols, density) in sizes {
        println!("\nTesting {} x {} with {:.3}% density", rows, cols, density * 100.0);
        
        let a = generate_large_sparse_matrix(rows, cols, density, 49);
        let b = generate_large_sparse_matrix(rows, cols, density, 50);
        
        println!("Matrix A: {} nnz", a.col_idx.len());
        println!("Matrix B: {} nnz", b.col_idx.len());
        
        let config = MagnusConfig::default();
        let start = std::time::Instant::now();
        
        let c = magnus_spgemm_parallel(&a, &b, &config);
        
        let elapsed = start.elapsed();
        println!("Multiplication took: {:.2}s", elapsed.as_secs_f64());
        println!("Result: {} nnz", c.col_idx.len());
        
        // Basic validation
        assert_eq!(c.n_rows, rows);
        assert_eq!(c.n_cols, cols);
    }
}