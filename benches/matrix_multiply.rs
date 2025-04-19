//! Benchmarks for sparse matrix multiplication

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use magnus::{SparseMatrixCSR, MagnusConfig};

/// This benchmark will be implemented as we develop the actual multiplication routines
fn bench_matrix_multiply(c: &mut Criterion) {
    // Create test matrices
    let a = create_test_matrix_a();
    let b = create_test_matrix_b();
    
    // This is a placeholder until we implement the actual multiplication
    c.bench_function("placeholder", |bench| {
        bench.iter(|| {
            // Just measure the overhead for now
            black_box(&a);
            black_box(&b);
        })
    });
    
    // Eventually, we'll add benchmarks for different multiplication strategies:
    // - Basic multiplication
    // - With fine-level reordering
    // - With coarse-level reordering
    // - With different accumulator thresholds
}

/// Create a test matrix for benchmarking
fn create_test_matrix_a() -> SparseMatrixCSR<f64> {
    // For now, just create a small test matrix
    // In the future, we'll implement more realistic test cases
    SparseMatrixCSR::new(
        100, 100,
        (0..=100).collect(),  // row_ptr - every row has 1 element (diagonal)
        (0..100).collect(),   // col_idx - diagonal elements
        vec![1.0; 100],       // values - all ones
    )
}

/// Create a test matrix for benchmarking
fn create_test_matrix_b() -> SparseMatrixCSR<f64> {
    // For now, just create a small test matrix
    SparseMatrixCSR::new(
        100, 100,
        (0..=100).collect(),  // row_ptr - every row has 1 element (diagonal)
        (0..100).collect(),   // col_idx - diagonal elements
        vec![2.0; 100],       // values - all twos
    )
}

criterion_group!(benches, bench_matrix_multiply);
criterion_main!(benches);