/// Tiered benchmark system for MAGNUS
///
/// Usage:
///   Quick sanity check (30s):     BENCH_TIER=quick cargo bench --bench tiered_benchmark
///   Standard tests (2-3 min):     BENCH_TIER=standard cargo bench --bench tiered_benchmark  
///   Large matrices only (5 min):  BENCH_TIER=large cargo bench --bench tiered_benchmark
///   Everything (10+ min):         BENCH_TIER=full cargo bench --bench tiered_benchmark
///
/// The quick tier is designed to catch major regressions in under a minute.
/// The large tier focuses exclusively on the primary use case (>1M nnz matrices).
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use magnus::{magnus_spgemm, magnus_spgemm_parallel, MagnusConfig, SparseMatrixCSR};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Duration;

/// Get the benchmark tier from environment
fn get_bench_tier() -> String {
    std::env::var("BENCH_TIER").unwrap_or_else(|_| "quick".to_string())
}

/// Generate sparse matrix
fn generate_matrix(rows: usize, cols: usize, nnz_per_row: usize) -> SparseMatrixCSR<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx = Vec::with_capacity(rows * nnz_per_row);
    let mut values = Vec::with_capacity(rows * nnz_per_row);

    row_ptr.push(0);

    for _ in 0..rows {
        let row_nnz = rng
            .gen_range((nnz_per_row * 3 / 4)..=(nnz_per_row * 5 / 4))
            .min(cols);
        let mut row_cols = std::collections::BTreeSet::new();

        for _ in 0..row_nnz {
            row_cols.insert(rng.gen_range(0..cols));
        }

        for col in row_cols {
            col_idx.push(col);
            values.push(rng.gen_range(0.1..10.0));
        }

        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

/// Quick tier - basic sanity checks, <1 minute total
fn bench_tier_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier_quick");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    println!("\n=== Running QUICK benchmark tier (sanity check) ===\n");

    // Very small matrix - catches major algorithmic breaks
    let tiny = generate_matrix(500, 500, 20);
    group.bench_function("500x500_serial", |b| {
        let config = MagnusConfig::default();
        b.iter(|| magnus_spgemm(black_box(&tiny), black_box(&tiny), &config));
    });

    // Small matrix - basic performance check
    let small = generate_matrix(2000, 2000, 50);
    group.bench_function("2k_parallel", |b| {
        let config = MagnusConfig::default();
        b.iter(|| magnus_spgemm_parallel(black_box(&small), black_box(&small), &config));
    });

    // Medium matrix - parallel vs serial comparison
    let medium = generate_matrix(5000, 5000, 100);
    let nnz = medium.col_idx.len();

    group.bench_function(format!("5k_serial_{:.1}M", nnz as f64 / 1e6), |b| {
        let config = MagnusConfig::default();
        b.iter(|| magnus_spgemm(black_box(&medium), black_box(&medium), &config));
    });

    group.bench_function(format!("5k_parallel_{:.1}M", nnz as f64 / 1e6), |b| {
        let config = MagnusConfig::default();
        b.iter(|| magnus_spgemm_parallel(black_box(&medium), black_box(&medium), &config));
    });

    group.finish();
    println!("\n=== Quick tier complete ===\n");
}

/// Standard tier - comprehensive test of normal use cases
fn bench_tier_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier_standard");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    println!("\n=== Running STANDARD benchmark tier ===\n");

    // Range of matrix sizes
    let test_cases = vec![
        (5_000, 100),  // 0.5M nnz
        (10_000, 100), // 1M nnz
        (15_000, 100), // 1.5M nnz
    ];

    for (size, nnz_per_row) in test_cases {
        let matrix = generate_matrix(size, size, nnz_per_row);
        let nnz = matrix.col_idx.len();

        group.bench_with_input(
            BenchmarkId::new(
                "standard",
                format!("{}x{}_{:.1}M", size, size, nnz as f64 / 1e6),
            ),
            &matrix,
            |b, m| {
                let config = MagnusConfig::default();
                b.iter(|| magnus_spgemm_parallel(black_box(m), black_box(m), &config));
            },
        );
    }

    group.finish();
    println!("\n=== Standard tier complete ===\n");
}

/// Large tier - focus on large sparse matrices (primary use case)
fn bench_tier_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier_large");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    println!("\n=== Running LARGE benchmark tier (primary use case) ===\n");

    // Large matrices with varying characteristics
    let test_cases = vec![
        // (name, size, nnz_per_row)
        ("dense_rows", 15_000, 150),    // ~2.25M nnz, dense rows
        ("medium_density", 30_000, 75), // ~2.25M nnz, medium density
        ("sparse", 50_000, 50),         // ~2.5M nnz, sparse
        ("very_sparse", 100_000, 30),   // ~3M nnz, very sparse
        ("huge_sparse", 150_000, 20),   // ~3M nnz, huge and sparse
    ];

    for (name, size, nnz_per_row) in test_cases {
        let matrix = generate_matrix(size, size, nnz_per_row);
        let nnz = matrix.col_idx.len();
        let density = nnz as f64 / (size * size) as f64 * 100.0;

        println!(
            "  Testing {}: {}x{} matrix, {:.2}M nnz, {:.4}% density",
            name,
            size,
            size,
            nnz as f64 / 1e6,
            density
        );

        group.bench_with_input(
            BenchmarkId::new("large", format!("{}_{:.1}M", name, nnz as f64 / 1e6)),
            &matrix,
            |b, m| {
                let config = MagnusConfig::default();
                b.iter(|| magnus_spgemm_parallel(black_box(m), black_box(m), &config));
            },
        );
    }

    group.finish();
    println!("\n=== Large tier complete ===\n");
}

/// Full tier - everything including stress tests
fn bench_tier_full(c: &mut Criterion) {
    // Run all other tiers first
    bench_tier_quick(c);
    bench_tier_standard(c);
    bench_tier_large(c);

    // Additional stress tests
    let mut group = c.benchmark_group("tier_stress");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);

    println!("\n=== Running STRESS tests (full tier) ===\n");

    // Extreme cases
    let test_cases = vec![
        ("10M_nnz", 200_000, 50), // ~10M nnz
        ("20M_nnz", 300_000, 67), // ~20M nnz
    ];

    for (name, size, nnz_per_row) in test_cases {
        let matrix = generate_matrix(size, size, nnz_per_row);
        let nnz = matrix.col_idx.len();

        println!(
            "  Stress test {}: {}x{} matrix, {:.1}M nnz",
            name,
            size,
            size,
            nnz as f64 / 1e6
        );

        group.bench_with_input(BenchmarkId::new("stress", name), &matrix, |b, m| {
            let config = MagnusConfig::default();
            b.iter(|| magnus_spgemm_parallel(black_box(m), black_box(m), &config));
        });
    }

    group.finish();
    println!("\n=== Full tier complete ===\n");
}

/// Main benchmark runner that selects tier based on environment
fn run_tiered_benchmarks(c: &mut Criterion) {
    let tier = get_bench_tier();

    println!("\n========================================");
    println!("  MAGNUS Tiered Benchmark System");
    println!("  Selected tier: {}", tier.to_uppercase());
    println!("========================================");

    match tier.as_str() {
        "quick" => bench_tier_quick(c),
        "standard" => bench_tier_standard(c),
        "large" => bench_tier_large(c),
        "full" => bench_tier_full(c),
        _ => {
            println!("Unknown tier '{}', defaulting to 'quick'", tier);
            bench_tier_quick(c);
        }
    }

    println!("\n========================================");
    println!("  Benchmark complete!");
    println!("  Tier: {}", tier.to_uppercase());
    println!("========================================\n");
}

criterion_group!(benches, run_tiered_benchmarks);
criterion_main!(benches);
