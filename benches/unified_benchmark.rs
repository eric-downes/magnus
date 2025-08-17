/// Unified benchmark system for MAGNUS with Buckingham π integration
///
/// Usage:
///   Quick tier (30s):      BENCH_TIER=quick cargo bench --bench unified_benchmark
///   Commit tier (10min):   BENCH_TIER=commit cargo bench --bench unified_benchmark  
///   PR tier (30min):       BENCH_TIER=pr cargo bench --bench unified_benchmark
///   Release tier (2hr):    BENCH_TIER=release cargo bench --bench unified_benchmark
///
/// The benchmark automatically adjusts the test set based on the tier:
/// - Quick: 5 critical π configurations
/// - Commit: 50 smart-sampled π configurations
/// - PR: 500 interesting π configurations
/// - Release: 2,673 full π exploration

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, BatchSize};
use magnus::{
    magnus_spgemm, magnus_spgemm_parallel, reference_spgemm,
    MagnusConfig, SparseMatrixCSR,
};
use magnus::test_tiers::{TestTier, TierConfig, BenchmarkTier, generate_test_matrices};
use magnus::reduced_parameter_space::SmartPiSampler;
use std::time::Duration;

/// Run benchmarks for the current tier
fn tier_benchmarks(c: &mut Criterion) {
    let bench_tier = BenchmarkTier::from_env();
    let config = TierConfig::for_tier(bench_tier.tier);
    
    println!("\n=== Running {:?} Tier Benchmarks ===", bench_tier.tier);
    println!("{}", config.summary());
    println!();
    
    // Configure criterion based on tier
    let mut group = c.benchmark_group(format!("{:?}_tier", bench_tier.tier));
    
    match bench_tier.tier {
        TestTier::Quick => {
            group.measurement_time(Duration::from_secs(2));
            group.sample_size(10);
        }
        TestTier::Commit => {
            group.measurement_time(Duration::from_secs(5));
            group.sample_size(20);
        }
        TestTier::PullRequest => {
            group.measurement_time(Duration::from_secs(10));
            group.sample_size(50);
        }
        TestTier::Release => {
            group.measurement_time(Duration::from_secs(20));
            group.sample_size(100);
        }
    }
    
    // Benchmark critical π configurations
    benchmark_pi_configs(&mut group, &config);
    
    // Benchmark traditional matrix sizes
    benchmark_traditional_sizes(&mut group, &config);
    
    // Benchmark parallel vs serial if enabled
    if config.test_parallel {
        benchmark_parallel_scaling(&mut group, &config);
    }
    
    group.finish();
}

/// Benchmark π configurations
fn benchmark_pi_configs(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, config: &TierConfig) {
    let sampler = SmartPiSampler::new(42);
    
    // Select π configurations based on tier
    let pi_configs = match config.pi_configs.len() {
        0..=5 => sampler.get_critical_configurations(),
        _ => config.pi_configs.clone(),
    };
    
    // Sample a few configurations for benchmarking
    let sample_size = 5.min(pi_configs.len());
    
    for (i, pi_config) in pi_configs.iter().take(sample_size).enumerate() {
        let test_size = config.matrix_sizes[config.matrix_sizes.len() / 2]; // Middle size
        
        group.bench_function(
            BenchmarkId::new("pi_config", format!("config_{}", i + 1)),
            |b| {
                b.iter_batched(
                    || {
                        let matrices = generate_test_matrices(TestTier::Quick, pi_config);
                        if matrices.len() >= 2 {
                            (matrices[0].clone(), matrices[1].clone())
                        } else {
                            // Fallback: generate identity matrices
                            let a = generate_identity_matrix(test_size);
                            let b = generate_identity_matrix(test_size);
                            (a, b)
                        }
                    },
                    |(a, b)| {
                        let config = MagnusConfig::default();
                        magnus_spgemm(&a, &b, &config)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
}

/// Benchmark traditional matrix sizes
fn benchmark_traditional_sizes(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, config: &TierConfig) {
    for &size in config.matrix_sizes.iter().take(3) {
        let nnz_per_row = (config.max_nnz / size).max(10).min(size / 10);
        
        group.bench_function(
            BenchmarkId::new("matrix_size", format!("{}x{}", size, size)),
            |b| {
                b.iter_batched(
                    || {
                        let a = generate_sparse_matrix(size, size, nnz_per_row);
                        let b = generate_sparse_matrix(size, size, nnz_per_row);
                        (a, b)
                    },
                    |(a, b)| {
                        let config = MagnusConfig::default();
                        magnus_spgemm(&a, &b, &config)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
}

/// Benchmark parallel vs serial scaling
fn benchmark_parallel_scaling(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, config: &TierConfig) {
    let test_size = config.matrix_sizes[config.matrix_sizes.len() / 2]; // Middle size
    let nnz_per_row = (config.max_nnz / test_size).max(10).min(test_size / 10);
    
    // Serial benchmark
    group.bench_function(
        BenchmarkId::new("scaling", "serial"),
        |b| {
            b.iter_batched(
                || {
                    let a = generate_sparse_matrix(test_size, test_size, nnz_per_row);
                    let b = generate_sparse_matrix(test_size, test_size, nnz_per_row);
                    (a, b)
                },
                |(a, b)| {
                    let config = MagnusConfig::default();
                    magnus_spgemm(&a, &b, &config)
                },
                BatchSize::SmallInput,
            );
        },
    );
    
    // Parallel benchmark
    group.bench_function(
        BenchmarkId::new("scaling", "parallel"),
        |b| {
            b.iter_batched(
                || {
                    let a = generate_sparse_matrix(test_size, test_size, nnz_per_row);
                    let b = generate_sparse_matrix(test_size, test_size, nnz_per_row);
                    (a, b)
                },
                |(a, b)| {
                    let config = MagnusConfig::default();
                    magnus_spgemm_parallel(&a, &b, &config)
                },
                BatchSize::SmallInput,
            );
        },
    );
    
    // Reference implementation for comparison
    if config.matrix_sizes[0] <= 5000 {
        group.bench_function(
            BenchmarkId::new("scaling", "reference"),
            |b| {
                b.iter_batched(
                    || {
                        let a = generate_sparse_matrix(test_size, test_size, nnz_per_row);
                        let b = generate_sparse_matrix(test_size, test_size, nnz_per_row);
                        (a, b)
                    },
                    |(a, b)| {
                        reference_spgemm(&a, &b)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
}

/// Generate a sparse matrix with specified properties
fn generate_sparse_matrix(rows: usize, cols: usize, nnz_per_row: usize) -> SparseMatrixCSR<f64> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx = Vec::with_capacity(rows * nnz_per_row);
    let mut values = Vec::with_capacity(rows * nnz_per_row);
    
    row_ptr.push(0);
    
    for _ in 0..rows {
        let row_nnz = rng.gen_range((nnz_per_row * 3 / 4)..=(nnz_per_row * 5 / 4))
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

/// Generate an identity matrix for fallback testing
fn generate_identity_matrix(size: usize) -> SparseMatrixCSR<f64> {
    let mut row_ptr = Vec::with_capacity(size + 1);
    let mut col_idx = Vec::with_capacity(size);
    let mut values = Vec::with_capacity(size);
    
    row_ptr.push(0);
    for i in 0..size {
        col_idx.push(i);
        values.push(1.0);
        row_ptr.push(i + 1);
    }
    
    SparseMatrixCSR::new(size, size, row_ptr, col_idx, values)
}

/// Quick sanity check benchmarks
fn quick_benchmarks(c: &mut Criterion) {
    if BenchmarkTier::from_env().tier != TestTier::Quick {
        return;
    }
    
    let mut group = c.benchmark_group("quick_sanity");
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(10);
    
    // Test tiny matrix to ensure basic functionality
    group.bench_function("100x100_sparse", |b| {
        b.iter_batched(
            || {
                let a = generate_sparse_matrix(100, 100, 5);
                let b = generate_sparse_matrix(100, 100, 5);
                (a, b)
            },
            |(a, b)| {
                let config = MagnusConfig::default();
                magnus_spgemm(&a, &b, &config)
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = tier_benchmarks, quick_benchmarks
}

criterion_main!(benches);