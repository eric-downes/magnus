//! Benchmarks for sparse matrix multiplication

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use magnus::{
    magnus_spgemm, magnus_spgemm_parallel, multiply_row_coarse_level, multiply_row_dense,
    multiply_row_fine_level, multiply_row_sort, process_coarse_level_rows_parallel,
    reference_spgemm, reordering, MagnusConfig, SparseMatrixCSR,
};
use std::hint::black_box;

/// Benchmark row multiplication operations
fn bench_row_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("Row Multiply Operations");

    let config = MagnusConfig::default();

    // Test with different row sizes (small, medium, large)
    let sizes = [100, 500, 1000];

    for &size in &sizes {
        // Create matrix B (size x size)
        let b = create_test_matrix_b(size);

        // Create test matrices with sparse rows
        let a_sparse = create_matrix_with_test_row(size, 0.1); // 10% density
        let a_medium = create_matrix_with_test_row(size, 0.2); // 20% density
        let a_dense = create_matrix_with_test_row(size, 0.5); // 50% density

        // Benchmark reference implementation
        group.bench_with_input(BenchmarkId::new("reference", size), &size, |bench, _| {
            bench.iter(|| {
                let c = reference_spgemm(&a_sparse, &b);
                black_box(c)
            })
        });

        // Benchmark dense accumulator
        group.bench_with_input(
            BenchmarkId::new("dense_accumulator", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let (cols, vals) = multiply_row_dense(0, &a_sparse, &b);
                    black_box((cols, vals))
                })
            },
        );

        // Benchmark sort-based accumulator
        group.bench_with_input(
            BenchmarkId::new("sort_accumulator", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let (cols, vals) = multiply_row_sort(0, &a_sparse, &b);
                    black_box((cols, vals))
                })
            },
        );

        // Benchmark fine-level reordering
        group.bench_with_input(BenchmarkId::new("fine_level", size), &size, |bench, _| {
            bench.iter(|| {
                let (cols, vals) = multiply_row_fine_level(0, &a_medium, &b, &config);
                black_box((cols, vals))
            })
        });

        // Benchmark coarse-level reordering
        group.bench_with_input(BenchmarkId::new("coarse_level", size), &size, |bench, _| {
            bench.iter(|| {
                let (cols, vals) = multiply_row_coarse_level(0, &a_dense, &b, &config);
                black_box((cols, vals))
            })
        });
    }

    group.finish();
}

/// Benchmark coarse-level processing
fn bench_coarse_level_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Coarse Level Processing");

    // Create test matrices of different sizes
    let sizes = [100, 500];

    for &size in &sizes {
        // Create dense matrices for coarse-level testing
        let a = create_random_matrix(size, size, 0.5); // Higher density
        let b = create_random_matrix(size, size, 0.5);

        // Create a sample of rows to process
        let rows: Vec<usize> = (0..std::cmp::min(size, 20)).collect();

        let config = MagnusConfig::default();

        // Benchmark sequential coarse-level processing
        group.bench_with_input(
            BenchmarkId::new(format!("coarse_sequential_{}", size), size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let results = reordering::process_coarse_level_rows(&a, &b, &rows, &config);
                    black_box(results)
                })
            },
        );

        // Benchmark parallel coarse-level processing
        group.bench_with_input(
            BenchmarkId::new(format!("coarse_parallel_{}", size), size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let results = process_coarse_level_rows_parallel(&a, &b, &rows, &config);
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark matrix generation for synthetic test cases
fn bench_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Multiply");

    // Create test matrices of different sizes
    let sizes = [100, 500]; // We'll use smaller sizes for faster testing
    let densities = [0.01, 0.05]; // 1%, 5% density

    for &size in &sizes {
        for &density in &densities {
            let a = create_random_matrix(size, size, density);
            let b = create_random_matrix(size, size, density);

            let config = MagnusConfig::default();

            // Benchmark reference multiplication
            group.bench_with_input(
                BenchmarkId::new(
                    format!("reference_{}_{}", size, (density * 100.0) as u32),
                    size,
                ),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        let c = reference_spgemm(&a, &b);
                        black_box(c)
                    })
                },
            );

            // Benchmark sequential MAGNUS implementation
            group.bench_with_input(
                BenchmarkId::new(
                    format!("magnus_{}_{}", size, (density * 100.0) as u32),
                    size,
                ),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        let c = magnus_spgemm(&a, &b, &config);
                        black_box(c)
                    })
                },
            );

            // Benchmark parallel MAGNUS implementation
            group.bench_with_input(
                BenchmarkId::new(
                    format!("magnus_parallel_{}_{}", size, (density * 100.0) as u32),
                    size,
                ),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        let c = magnus_spgemm_parallel(&a, &b, &config);
                        black_box(c)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Create a test matrix for benchmarking
fn create_test_matrix_a(size: usize) -> SparseMatrixCSR<f64> {
    // More realistic test matrix (sparse tridiagonal)
    let n = size;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(3 * n);
    let mut values = Vec::with_capacity(3 * n);

    row_ptr.push(0);
    let mut nnz = 0;

    for i in 0..n {
        // Add lower diagonal (except first row)
        if i > 0 {
            col_idx.push(i - 1);
            values.push(1.0);
            nnz += 1;
        }

        // Add diagonal
        col_idx.push(i);
        values.push(2.0);
        nnz += 1;

        // Add upper diagonal (except last row)
        if i < n - 1 {
            col_idx.push(i + 1);
            values.push(1.0);
            nnz += 1;
        }

        row_ptr.push(nnz);
    }

    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

/// Create a test matrix for benchmarking
fn create_test_matrix_b(size: usize) -> SparseMatrixCSR<f64> {
    // Similar structure to matrix A
    create_test_matrix_a(size)
}

/// Create a random sparse matrix with given dimensions and density
fn create_random_matrix(rows: usize, cols: usize, density: f64) -> SparseMatrixCSR<f64> {
    let mut row_ptr = Vec::with_capacity(rows + 1);
    let expected_nnz = (rows as f64 * cols as f64 * density) as usize;
    let mut col_idx = Vec::with_capacity(expected_nnz);
    let mut values = Vec::with_capacity(expected_nnz);

    row_ptr.push(0);
    let mut nnz = 0;

    for i in 0..rows {
        // For each row, add nnz elements based on density
        let row_nnz = (cols as f64 * density) as usize;

        // Simple deterministic "random" pattern
        for j in 0..row_nnz {
            let col = (i + j * 7) % cols; // Pseudo-random distribution
            col_idx.push(col);
            values.push(1.0 + (i % 10) as f64 / 10.0); // Some variation in values
            nnz += 1;
        }

        row_ptr.push(nnz);
    }

    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

/// Create a test row with given density
fn create_test_row(cols: usize, density: f64) -> (Vec<usize>, Vec<f64>) {
    let nnz = (cols as f64 * density) as usize;
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    // Create a deterministic pattern
    for j in 0..nnz {
        let col = (j * cols / nnz) % cols; // Distribute across columns
        col_idx.push(col);
        values.push(1.0 + (j % 5) as f64 / 5.0); // Some variation in values
    }

    (col_idx, values)
}

/// Create a test matrix with a single non-zero row
fn create_matrix_with_test_row(cols: usize, density: f64) -> SparseMatrixCSR<f64> {
    let (col_indices, values) = create_test_row(cols, density);
    let nnz = col_indices.len();

    // Create CSR format with a single row
    let row_ptr = vec![0, nnz];

    SparseMatrixCSR::new(1, cols, row_ptr, col_indices, values)
}

criterion_group!(
    benches,
    bench_row_multiply,
    bench_matrix_multiply,
    bench_coarse_level_processing
);
criterion_main!(benches);
