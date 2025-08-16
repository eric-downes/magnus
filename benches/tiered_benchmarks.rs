//! Tiered benchmark system for MAGNUS
//!
//! Three levels of benchmarks:
//! 1. QUICK (< 30 seconds) - Sanity checks, small matrices
//! 2. STANDARD (< 5 minutes) - Medium matrices, parameter tuning
//! 3. STRESS (10+ minutes) - Large matrices, real-world workloads

use criterion::{criterion_group, BenchmarkId, Criterion};
use magnus::{magnus_spgemm, magnus_spgemm_parallel, MagnusConfig, SparseMatrixCSR};
use std::hint::black_box;
use std::time::Instant;

/// Generate a sparse matrix with realistic structure
fn generate_realistic_sparse_matrix(n: usize, nnz_per_row: usize) -> SparseMatrixCSR<f64> {
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42); // Deterministic for reproducibility

    for i in 0..n {
        let mut cols = std::collections::HashSet::new();

        // Add some structure: diagonal and near-diagonal entries
        if i > 0 {
            cols.insert(i - 1);
        }
        cols.insert(i);
        if i < n - 1 {
            cols.insert(i + 1);
        }

        // Add random entries
        while cols.len() < nnz_per_row.min(n) {
            cols.insert(rng.gen_range(0..n));
        }

        let mut sorted_cols: Vec<_> = cols.into_iter().collect();
        sorted_cols.sort();

        for col in sorted_cols {
            col_idx.push(col);
            values.push(1.0 + rng.gen::<f64>());
        }

        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

/// Pre-flight sanity check - ensures basic functionality before expensive benchmarks
fn preflight_check() -> Result<(), String> {
    println!("🔍 Running pre-flight sanity checks...");

    // Test 1: Very small matrix
    let tiny_a = generate_realistic_sparse_matrix(10, 3);
    let tiny_b = generate_realistic_sparse_matrix(10, 3);
    let config = MagnusConfig::default();

    let start = Instant::now();
    let result = magnus_spgemm(&tiny_a, &tiny_b, &config);
    let duration = start.elapsed();

    if result.n_rows != 10 || result.n_cols != 10 {
        return Err(format!("Tiny matrix test failed: wrong dimensions"));
    }

    println!("  ✅ Tiny matrix (10x10): {:?}", duration);

    // Test 2: Small matrix
    let small_a = generate_realistic_sparse_matrix(100, 10);
    let small_b = generate_realistic_sparse_matrix(100, 10);

    let start = Instant::now();
    let result = magnus_spgemm(&small_a, &small_b, &config);
    let duration = start.elapsed();

    if result.n_rows != 100 || result.n_cols != 100 {
        return Err(format!("Small matrix test failed: wrong dimensions"));
    }

    println!("  ✅ Small matrix (100x100): {:?}", duration);

    // Test 3: Check parallel execution
    let start = Instant::now();
    let result_parallel = magnus_spgemm_parallel(&small_a, &small_b, &config);
    let duration = start.elapsed();

    if result_parallel.n_rows != 100 || result_parallel.n_cols != 100 {
        return Err(format!("Parallel test failed: wrong dimensions"));
    }

    println!("  ✅ Parallel execution: {:?}", duration);

    // Test 4: Check if times are reasonable
    if duration > std::time::Duration::from_secs(5) {
        return Err(format!(
            "Performance issue detected: small matrix took {:?}",
            duration
        ));
    }

    println!("✅ All pre-flight checks passed!\n");
    Ok(())
}

/// TIER 1: Quick benchmarks (< 30 seconds total)
/// Used for rapid iteration and CI/CD
fn bench_tier1_quick(c: &mut Criterion) {
    // Run sanity checks first
    if let Err(e) = preflight_check() {
        panic!("Pre-flight check failed: {}", e);
    }

    let mut group = c.benchmark_group("tier1_quick");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    println!("📊 TIER 1: Quick Benchmarks (matrices up to 500x500)");

    let sizes = vec![
        (50, 5),   // 50x50, 5 nnz/row
        (100, 10), // 100x100, 10 nnz/row
        (200, 15), // 200x200, 15 nnz/row
        (500, 20), // 500x500, 20 nnz/row
    ];

    for (size, nnz) in sizes {
        let a = generate_realistic_sparse_matrix(size, nnz);
        let b = generate_realistic_sparse_matrix(size, nnz);
        let config = MagnusConfig::default();

        group.bench_with_input(
            BenchmarkId::new("SpGEMM", format!("{}x{}_nnz{}", size, size, nnz)),
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

/// TIER 2: Standard benchmarks (< 5 minutes total)
/// Used for performance validation and regression testing
fn bench_tier2_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier2_standard");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    println!("📊 TIER 2: Standard Benchmarks (matrices up to 5000x5000)");

    let sizes = vec![
        (1000, 50),  // 1K x 1K, 50 nnz/row
        (2000, 100), // 2K x 2K, 100 nnz/row
        (5000, 200), // 5K x 5K, 200 nnz/row
    ];

    for (size, nnz) in sizes {
        println!(
            "  Generating {}x{} matrix with {} nnz/row...",
            size, size, nnz
        );
        let a = generate_realistic_sparse_matrix(size, nnz);
        let b = generate_realistic_sparse_matrix(size, nnz);
        let config = MagnusConfig::default();

        // Serial execution
        group.bench_with_input(
            BenchmarkId::new("Serial", format!("{}x{}_nnz{}", size, size, nnz)),
            &(&a, &b, &config),
            |bench, (a, b, config)| {
                bench.iter(|| {
                    let c = magnus_spgemm(a, b, config);
                    black_box(c)
                })
            },
        );

        // Parallel execution
        group.bench_with_input(
            BenchmarkId::new("Parallel", format!("{}x{}_nnz{}", size, size, nnz)),
            &(&a, &b, &config),
            |bench, (a, b, config)| {
                bench.iter(|| {
                    let c = magnus_spgemm_parallel(a, b, config);
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}

/// TIER 3: Stress tests (10+ minutes)
/// Used for finding performance limits and real-world validation
fn bench_tier3_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier3_stress");
    group.sample_size(10); // Few samples for large matrices
    group.measurement_time(std::time::Duration::from_secs(30));

    println!("📊 TIER 3: Stress Tests (large matrices, may take 10+ minutes)");
    println!("  ⚠️  These benchmarks will consume significant memory and time");

    let sizes = vec![
        (10_000, 500),  // 10K x 10K, 500 nnz/row (~5M nonzeros)
        (50_000, 1000), // 50K x 50K, 1000 nnz/row (~50M nonzeros)
        (100_000, 100), // 100K x 100K, 100 nnz/row (~10M nonzeros, sparse)
    ];

    for (size, nnz) in sizes {
        println!(
            "\n  🔨 Stress test: {}x{} matrix with {} nnz/row",
            size, size, nnz
        );
        println!("    Estimated memory: {} MB", (size * nnz * 16) / 1_000_000);

        let start = Instant::now();
        println!("    Generating matrices...");
        let a = generate_realistic_sparse_matrix(size, nnz);
        let b = generate_realistic_sparse_matrix(size, nnz);
        println!("    Matrix generation took: {:?}", start.elapsed());

        let config = MagnusConfig::default();

        // Only test parallel for large matrices (serial would be too slow)
        group.bench_with_input(
            BenchmarkId::new("Parallel", format!("{}x{}_nnz{}", size, size, nnz)),
            &(&a, &b, &config),
            |bench, (a, b, config)| {
                bench.iter(|| {
                    let c = magnus_spgemm_parallel(a, b, config);
                    black_box(c)
                })
            },
        );

        println!("    ✅ Completed {}x{} stress test", size, size);
    }

    group.finish();
}

// Define benchmark groups based on feature flags or environment variables
criterion_group! {
    name = quick;
    config = Criterion::default();
    targets = bench_tier1_quick
}

criterion_group! {
    name = standard;
    config = Criterion::default();
    targets = bench_tier2_standard
}

criterion_group! {
    name = stress;
    config = Criterion::default();
    targets = bench_tier3_stress
}

// Main entry point - selects which tier to run based on environment
fn main() {
    let tier = std::env::var("BENCH_TIER").unwrap_or_else(|_| "quick".to_string());

    println!("\n🚀 MAGNUS Tiered Benchmark System");
    println!("=====================================");

    match tier.as_str() {
        "quick" | "1" => {
            println!("Running TIER 1 (Quick) benchmarks...\n");
            quick();
        }
        "standard" | "2" => {
            println!("Running TIER 2 (Standard) benchmarks...\n");
            println!("⚠️  This will take several minutes\n");
            standard();
        }
        "stress" | "3" => {
            println!("Running TIER 3 (Stress) benchmarks...\n");
            println!("⚠️  WARNING: This will take 10+ minutes and use significant memory!\n");
            println!("Press Ctrl+C to cancel if not intended.\n");
            std::thread::sleep(std::time::Duration::from_secs(3));
            stress();
        }
        "all" => {
            println!("Running ALL tiers (this will take a long time)...\n");
            quick();
            standard();
            stress();
        }
        _ => {
            eprintln!("Unknown tier: {}", tier);
            eprintln!("Usage: BENCH_TIER=<quick|standard|stress|all> cargo bench --bench tiered_benchmarks");
            std::process::exit(1);
        }
    }
}
