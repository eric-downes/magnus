//! Benchmark comparing MAGNUS against standard Rust sparse matrix libraries
//!
//! This provides a principled comparison of MAGNUS SpGEMM performance against
//! the sprs crate, which is the standard Rust sparse matrix library.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use magnus::{magnus_spgemm, MagnusConfig, SparseMatrixCSR};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use std::hint::black_box;
use std::time::Duration;

/// Generate a test matrix with specified dimensions and density
fn generate_test_matrix(rows: usize, cols: usize, density: f64, seed: u64) -> SparseMatrixCSR<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let nnz_target = ((rows * cols) as f64 * density) as usize;

    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    for _row in 0..rows {
        let row_nnz = rng.gen_range(0..=(2 * nnz_target / rows).min(cols));
        let mut cols_in_row: Vec<usize> = (0..cols).collect();
        cols_in_row.shuffle(&mut rng);
        cols_in_row.truncate(row_nnz);
        cols_in_row.sort_unstable();

        for col in cols_in_row {
            col_idx.push(col);
            values.push(rng.gen::<f64>() * 10.0);
        }
        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

/// Convert MAGNUS matrix to sprs format
fn to_sprs(matrix: &SparseMatrixCSR<f64>) -> sprs::CsMat<f64> {
    // Create from triplets to ensure proper sorting and deduplication
    let mut row_inds = Vec::new();
    let mut col_inds = Vec::new();
    let mut vals = Vec::new();

    for i in 0..matrix.n_rows {
        let start = matrix.row_ptr[i];
        let end = matrix.row_ptr[i + 1];
        for j in start..end {
            row_inds.push(i);
            col_inds.push(matrix.col_idx[j]);
            vals.push(matrix.values[j]);
        }
    }

    sprs::TriMat::from_triplets((matrix.n_rows, matrix.n_cols), row_inds, col_inds, vals).to_csr()
}

/// Benchmark group comparing small matrices (< 1000 non-zeros)
fn bench_small_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_matrices_comparison");
    group.sample_size(50); // Reduce sample size for consistency
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [10, 50, 100] {
        let density = 0.1;
        let a = generate_test_matrix(size, size, density, 42);
        let b = generate_test_matrix(size, size, density, 43);

        let a_sprs = to_sprs(&a);
        let b_sprs = to_sprs(&b);

        let nnz = a.values.len() + b.values.len();
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_function(BenchmarkId::new("MAGNUS", size), |bencher| {
            bencher.iter(|| {
                let config = MagnusConfig::default();
                let _result = magnus_spgemm(black_box(&a), black_box(&b), black_box(&config));
            });
        });

        group.bench_function(BenchmarkId::new("sprs", size), |bencher| {
            bencher.iter(|| {
                let _result = black_box(&a_sprs) * black_box(&b_sprs);
            });
        });
    }
    group.finish();
}

/// Benchmark group comparing medium matrices (1K - 100K non-zeros)
fn bench_medium_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_matrices_comparison");
    group.sample_size(20); // Smaller sample size for larger matrices
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));

    for size in [200, 500, 1000] {
        let density = 0.02; // Lower density for larger matrices
        let a = generate_test_matrix(size, size, density, 42);
        let b = generate_test_matrix(size, size, density, 43);

        let a_sprs = to_sprs(&a);
        let b_sprs = to_sprs(&b);

        let nnz = a.values.len() + b.values.len();
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_function(BenchmarkId::new("MAGNUS", size), |bencher| {
            bencher.iter(|| {
                let config = MagnusConfig::default();
                let _result = magnus_spgemm(black_box(&a), black_box(&b), black_box(&config));
            });
        });

        group.bench_function(BenchmarkId::new("sprs", size), |bencher| {
            bencher.iter(|| {
                let _result = black_box(&a_sprs) * black_box(&b_sprs);
            });
        });
    }
    group.finish();
}

/// Benchmark group comparing different sparsity patterns
fn bench_sparsity_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity_patterns_comparison");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let size = 500;

    for (name, density) in [("sparse", 0.001), ("moderate", 0.01), ("dense", 0.1)] {
        let a = generate_test_matrix(size, size, density, 42);
        let b = generate_test_matrix(size, size, density, 43);

        let a_sprs = to_sprs(&a);
        let b_sprs = to_sprs(&b);

        let nnz = a.values.len() + b.values.len();
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_function(BenchmarkId::new("MAGNUS", name), |bencher| {
            bencher.iter(|| {
                let config = MagnusConfig::default();
                let _result = magnus_spgemm(black_box(&a), black_box(&b), black_box(&config));
            });
        });

        group.bench_function(BenchmarkId::new("sprs", name), |bencher| {
            bencher.iter(|| {
                let _result = black_box(&a_sprs) * black_box(&b_sprs);
            });
        });
    }
    group.finish();
}

/// Benchmark specific workloads where MAGNUS optimizations should excel
fn bench_magnus_optimized_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("magnus_optimized_cases");
    group.sample_size(20);

    // Case 1: Many duplicates (where sort-accumulate shines)
    let size = 300;
    let mut a = generate_test_matrix(size, size, 0.05, 42);

    // Artificially create duplicates by reducing column variety
    for idx in a.col_idx.iter_mut() {
        *idx = (*idx / 10) * 10; // Round to nearest 10
    }

    let b = a.clone(); // Square the matrix
    let a_sprs = to_sprs(&a);
    let b_sprs = to_sprs(&b);

    group.bench_function("MAGNUS_duplicates", |bencher| {
        bencher.iter(|| {
            let config = MagnusConfig::default();
            let _result = magnus_spgemm(black_box(&a), black_box(&b), black_box(&config));
        });
    });

    group.bench_function("sprs_duplicates", |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a_sprs) * black_box(&b_sprs);
        });
    });

    group.finish();
}

/// Print summary statistics about outliers
fn print_outlier_explanation() {
    println!("\n=== Understanding Benchmark Outliers ===\n");
    println!("Criterion.rs classifies outliers based on statistical analysis:");
    println!("- Mild: 1.5x IQR from quartiles (like box plot whiskers)");
    println!("- Severe: 3x IQR from quartiles");
    println!();
    println!("High outlier percentages usually indicate:");
    println!("1. System interference (background processes, CPU throttling)");
    println!("2. Memory effects (cache misses, page faults)");
    println!("3. Warmup issues (JIT, frequency scaling)");
    println!();
    println!("For sparse matrix operations, outliers are expected because:");
    println!("- Irregular memory access patterns");
    println!("- Variable work per matrix (depends on sparsity structure)");
    println!("- Cache effects vary with matrix size");
    println!();
    println!("The comparison benchmarks above provide the meaningful metric:");
    println!("MAGNUS performance relative to standard Rust sparse matrix library (sprs)");
    println!("\n=========================================\n");
}

// Quick tier benchmarks - run in < 30 seconds
criterion_group!(quick_benches, bench_small_matrices, bench_quick_comparison,);

// Full benchmarks - comprehensive comparison
criterion_group!(
    full_benches,
    bench_small_matrices,
    bench_medium_matrices,
    bench_sparsity_patterns,
    bench_magnus_optimized_cases,
);

/// Quick comparison for fast CI/development iteration
fn bench_quick_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_comparison");
    group.sample_size(10); // Very small sample for speed
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    // Just test a few representative sizes
    for size in [50, 200, 500] {
        let density = 0.02;
        let a = generate_test_matrix(size, size, density, 42);
        let b = generate_test_matrix(size, size, density, 43);

        let a_sprs = to_sprs(&a);
        let b_sprs = to_sprs(&b);

        group.bench_function(BenchmarkId::new("MAGNUS", size), |bencher| {
            let config = MagnusConfig::default();
            bencher.iter(|| {
                let _result = magnus_spgemm(black_box(&a), black_box(&b), black_box(&config));
            });
        });

        group.bench_function(BenchmarkId::new("sprs", size), |bencher| {
            bencher.iter(|| {
                let _result = black_box(&a_sprs) * black_box(&b_sprs);
            });
        });
    }
    group.finish();
}

fn main() {
    // Check for tier environment variable (similar to tiered_benchmarks.rs)
    let tier = std::env::var("BENCH_TIER").unwrap_or_else(|_| "full".to_string());

    match tier.as_str() {
        "quick" | "1" => {
            println!("Running QUICK comparison benchmarks (< 30 seconds)...\n");
            quick_benches();
        }
        "full" | "all" => {
            println!("Running FULL comparison benchmarks...\n");
            print_outlier_explanation();
            full_benches();
        }
        _ => {
            println!("Running FULL comparison benchmarks (default)...\n");
            print_outlier_explanation();
            full_benches();
        }
    }

    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
