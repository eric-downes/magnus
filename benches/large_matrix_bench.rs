/// Large sparse matrix benchmarks for MAGNUS
///
/// Run with: cargo bench --bench large_matrix_bench
/// For quick sanity check: BENCH_TIER=quick cargo bench --bench large_matrix_bench
/// For full large matrix suite: BENCH_TIER=large cargo bench --bench large_matrix_bench
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use magnus::{
    magnus_spgemm, magnus_spgemm_parallel, reference_spgemm, MagnusConfig, SparseMatrixCSR,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Duration;

/// Benchmark tier controlled by environment variable
fn get_bench_tier() -> String {
    std::env::var("BENCH_TIER").unwrap_or_else(|_| "quick".to_string())
}

/// Generate a sparse matrix for benchmarking
fn generate_bench_matrix(
    rows: usize,
    cols: usize,
    nnz_per_row: usize,
    seed: u64,
) -> SparseMatrixCSR<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx = Vec::with_capacity(rows * nnz_per_row);
    let mut values = Vec::with_capacity(rows * nnz_per_row);

    row_ptr.push(0);

    for _ in 0..rows {
        let row_nnz = rng
            .gen_range((nnz_per_row / 2)..=(nnz_per_row * 3 / 2))
            .min(cols);
        let mut row_cols = Vec::with_capacity(row_nnz);
        let mut used = std::collections::HashSet::new();

        for _ in 0..row_nnz {
            let mut col = rng.gen_range(0..cols);
            let mut attempts = 0;
            while used.contains(&col) && attempts < 100 {
                col = rng.gen_range(0..cols);
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

    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

/// Quick sanity check benchmarks - run fast to catch major regressions
fn bench_quick_sanity(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_sanity");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    // Small matrix for quick validation
    let small = generate_bench_matrix(1000, 1000, 50, 42);

    group.bench_function("magnus_1k", |b| {
        let config = MagnusConfig::default();
        b.iter(|| magnus_spgemm(black_box(&small), black_box(&small), &config));
    });

    group.bench_function("reference_1k", |b| {
        b.iter(|| reference_spgemm(black_box(&small), black_box(&small)));
    });

    // Medium matrix to check parallel vs serial
    let medium = generate_bench_matrix(5000, 5000, 100, 43);

    group.bench_function("magnus_parallel_5k", |b| {
        let config = MagnusConfig::default();
        b.iter(|| magnus_spgemm_parallel(black_box(&medium), black_box(&medium), &config));
    });

    group.finish();
}

/// Large matrix benchmarks - the main use case
fn bench_large_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_matrices");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    // Test different matrix sizes, all with >1M non-zeros
    let test_cases = vec![
        ("10k_dense", 10_000, 10_000, 100), // ~1M nnz, relatively dense rows
        ("20k_medium", 20_000, 20_000, 50), // ~1M nnz, medium density
        ("50k_sparse", 50_000, 50_000, 40), // ~2M nnz, sparser
        ("100k_ultra", 100_000, 100_000, 20), // ~2M nnz, very sparse
    ];

    for (name, rows, cols, nnz_per_row) in test_cases {
        let matrix = generate_bench_matrix(rows, cols, nnz_per_row, 44);
        let total_nnz = matrix.col_idx.len();

        group.bench_with_input(
            BenchmarkId::new(
                "magnus_parallel",
                format!("{}_{:.1}M", name, total_nnz as f64 / 1e6),
            ),
            &matrix,
            |b, m| {
                let config = MagnusConfig::default();
                b.iter(|| magnus_spgemm_parallel(black_box(m), black_box(m), &config));
            },
        );

        // Only benchmark serial version on smaller matrices
        if rows <= 20_000 {
            group.bench_with_input(
                BenchmarkId::new(
                    "magnus_serial",
                    format!("{}_{:.1}M", name, total_nnz as f64 / 1e6),
                ),
                &matrix,
                |b, m| {
                    let config = MagnusConfig::default();
                    b.iter(|| magnus_spgemm(black_box(m), black_box(m), &config));
                },
            );
        }
    }

    group.finish();
}

/// Scaling study - how performance scales with matrix size
fn bench_scaling_study(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_study");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    // Keep nnz/row constant, vary matrix size
    let sizes = vec![10_000, 20_000, 30_000, 40_000, 50_000];

    for size in sizes {
        let matrix = generate_bench_matrix(size, size, 50, 45);
        let total_nnz = matrix.col_idx.len();

        group.bench_with_input(
            BenchmarkId::new(
                "scaling",
                format!("{}x{}_{:.1}M", size, size, total_nnz as f64 / 1e6),
            ),
            &matrix,
            |b, m| {
                let config = MagnusConfig::default();
                b.iter(|| magnus_spgemm_parallel(black_box(m), black_box(m), &config));
            },
        );
    }

    group.finish();
}

/// Density study - how performance varies with matrix density
fn bench_density_study(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_study");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    // Fixed size, vary density
    let size = 25_000;
    let densities = vec![
        10,  // Very sparse
        25,  // Sparse
        50,  // Medium
        100, // Dense
        200, // Very dense
    ];

    for nnz_per_row in densities {
        let matrix = generate_bench_matrix(size, size, nnz_per_row, 46);
        let total_nnz = matrix.col_idx.len();
        let density = total_nnz as f64 / (size * size) as f64;

        group.bench_with_input(
            BenchmarkId::new(
                "density",
                format!("nnz{}_{:.3}%", nnz_per_row, density * 100.0),
            ),
            &matrix,
            |b, m| {
                let config = MagnusConfig::default();
                b.iter(|| magnus_spgemm_parallel(black_box(m), black_box(m), &config));
            },
        );
    }

    group.finish();
}

/// Configure benchmark groups based on tier
fn configure_criterion() -> Criterion {
    let tier = get_bench_tier();

    match tier.as_str() {
        "quick" => {
            // Quick sanity check - 30 seconds total
            Criterion::default()
                .measurement_time(Duration::from_secs(5))
                .sample_size(10)
        }
        "large" => {
            // Full large matrix suite - several minutes
            Criterion::default()
                .measurement_time(Duration::from_secs(30))
                .sample_size(10)
        }
        "full" => {
            // Everything including scaling studies
            Criterion::default()
                .measurement_time(Duration::from_secs(30))
                .sample_size(15)
        }
        _ => {
            // Default to quick
            Criterion::default()
                .measurement_time(Duration::from_secs(5))
                .sample_size(10)
        }
    }
}

// Select benchmark groups based on tier
criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_quick_sanity, bench_large_matrices, bench_scaling_study, bench_density_study
}

criterion_main!(benches);
