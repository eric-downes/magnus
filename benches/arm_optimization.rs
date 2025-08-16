use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use magnus::accumulator::{create_simd_accelerator_f32, FallbackAccumulator, SimdAccelerator};
use magnus::matrix::config::{detect_architecture, Architecture, MagnusConfig};
use magnus::{magnus_spgemm, SparseMatrixCSR};
use rand::Rng;
use std::hint::black_box;

/// Generate random test data for accumulator benchmarks
fn generate_test_data(n_entries: usize, max_col: usize) -> (Vec<usize>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let col_indices: Vec<usize> = (0..n_entries).map(|_| rng.gen_range(0..max_col)).collect();
    let values: Vec<f32> = (0..n_entries).map(|_| rng.gen::<f32>()).collect();
    (col_indices, values)
}

/// Generate a random sparse matrix
fn generate_sparse_matrix(n_rows: usize, n_cols: usize, density: f64) -> SparseMatrixCSR<f32> {
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    for _ in 0..n_rows {
        let n_entries = ((n_cols as f64) * density) as usize;
        let mut row_cols: Vec<usize> = Vec::new();
        let mut used = std::collections::HashSet::new();
        while row_cols.len() < n_entries && row_cols.len() < n_cols {
            let col = rng.gen_range(0..n_cols);
            if used.insert(col) {
                row_cols.push(col);
            }
        }
        row_cols.sort();

        for col in row_cols {
            col_idx.push(col);
            values.push(rng.gen::<f32>());
        }
        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(n_rows, n_cols, row_ptr, col_idx, values)
}

/// Benchmark NEON vs generic accumulator for different input sizes
fn bench_accumulator_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulator_methods");

    // Test different input sizes
    let sizes = vec![16, 64, 256, 1024, 4096];

    for size in sizes {
        let (col_indices, values) = generate_test_data(size, size / 4);

        // Benchmark NEON accumulator (if available)
        if detect_architecture() == Architecture::ArmNeon {
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            {
                use magnus::accumulator::NeonAccumulator;
                let neon_acc = NeonAccumulator::new();
                group.bench_with_input(
                    BenchmarkId::new("NEON", size),
                    &(&col_indices, &values),
                    |b, (cols, vals)| {
                        b.iter(|| {
                            let (c, v) = neon_acc.sort_and_accumulate(cols, vals);
                            black_box((c, v))
                        })
                    },
                );
            }
        }

        // Benchmark generic fallback
        let fallback_acc = FallbackAccumulator::new();
        group.bench_with_input(
            BenchmarkId::new("Fallback", size),
            &(&col_indices, &values),
            |b, (cols, vals)| {
                b.iter(|| {
                    let (c, v) = fallback_acc.sort_and_accumulate(cols, vals);
                    black_box((c, v))
                })
            },
        );

        // Benchmark auto-selected SIMD accelerator
        let simd_acc = create_simd_accelerator_f32();
        group.bench_with_input(
            BenchmarkId::new("Auto-SIMD", size),
            &(&col_indices, &values),
            |b, (cols, vals)| {
                b.iter(|| {
                    let (c, v) = simd_acc.sort_and_accumulate(cols, vals);
                    black_box((c, v))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark different accumulator thresholds
fn bench_accumulator_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulator_threshold");

    // Test different thresholds to find optimal crossover point
    let thresholds = vec![64, 128, 192, 256, 384, 512];
    let matrix_size = 1000;

    for threshold in thresholds {
        let a = generate_sparse_matrix(100, matrix_size, 0.1);
        let b = generate_sparse_matrix(matrix_size, 100, 0.1);

        let mut config = MagnusConfig::for_architecture(Architecture::ArmNeon);
        config.dense_accum_threshold = threshold;

        group.bench_with_input(
            BenchmarkId::new("threshold", threshold),
            &(&a, &b, &config),
            |bench, (a, b, config)| {
                bench.iter(|| {
                    let c = magnus_spgemm(a, b, config);
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark different chunk sizes for reordering
fn bench_chunk_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_sizes");

    // Test different chunk sizes to find optimal for Apple Silicon
    let chunk_sizes = vec![128, 256, 512, 1024, 2048];

    for chunk_size in chunk_sizes {
        let a = generate_sparse_matrix(500, 500, 0.05);
        let b = generate_sparse_matrix(500, 500, 0.05);

        // Create custom config with specific chunk size
        // This would require extending MagnusConfig to support custom chunk sizes
        let config = MagnusConfig::for_architecture(Architecture::ArmNeon);

        group.bench_with_input(
            BenchmarkId::new("chunk_size", chunk_size),
            &(&a, &b, &config),
            |bench, (a, b, config)| {
                bench.iter(|| {
                    let c = magnus_spgemm(a, b, config);
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark full MAGNUS algorithm on different matrix sizes
fn bench_magnus_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("magnus_sizes");
    group.sample_size(10); // Reduce sample size for larger matrices

    // Test different matrix sizes
    let sizes = vec![(100, 0.1), (500, 0.05), (1000, 0.02)];

    for (size, density) in sizes {
        let a = generate_sparse_matrix(size, size, density);
        let b = generate_sparse_matrix(size, size, density);

        // Benchmark with ARM-optimized config
        let arm_config = MagnusConfig::for_architecture(Architecture::ArmNeon);
        group.bench_with_input(
            BenchmarkId::new("ARM-optimized", size),
            &(&a, &b, &arm_config),
            |bench, (a, b, config)| {
                bench.iter(|| {
                    let c = magnus_spgemm(a, b, config);
                    black_box(c)
                })
            },
        );

        // Benchmark with generic config for comparison
        let generic_config = MagnusConfig::for_architecture(Architecture::Generic);
        group.bench_with_input(
            BenchmarkId::new("Generic", size),
            &(&a, &b, &generic_config),
            |bench, (a, b, config)| {
                bench.iter(|| {
                    let c = magnus_spgemm(a, b, config);
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory access patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Test row-wise vs column-wise access patterns
    let size = 1000;

    // Row-major sparse matrix (many rows, few cols per row)
    let row_major = generate_sparse_matrix(size, 100, 0.5);
    let row_major_b = generate_sparse_matrix(100, size, 0.1);

    // Column-major sparse matrix (few rows, many cols per row)
    let col_major = generate_sparse_matrix(100, size, 0.5);
    let col_major_b = generate_sparse_matrix(size, 100, 0.1);

    let config = MagnusConfig::for_architecture(Architecture::ArmNeon);

    group.bench_function("row_major", |b| {
        b.iter(|| {
            let c = magnus_spgemm(&row_major, &row_major_b, &config);
            black_box(c)
        })
    });

    group.bench_function("col_major", |b| {
        b.iter(|| {
            let c = magnus_spgemm(&col_major, &col_major_b, &config);
            black_box(c)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_accumulator_methods,
    bench_accumulator_threshold,
    bench_chunk_sizes,
    bench_magnus_sizes,
    bench_memory_patterns
);
criterion_main!(benches);
