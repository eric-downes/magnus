//! Benchmarks for memory prefetching optimization
//! 
//! This benchmark suite measures the impact of different prefetching strategies
//! on sparse matrix multiplication performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use magnus::{SparseMatrixCSR, magnus_spgemm, MagnusConfig};
use rand::Rng;
use std::time::Instant;

/// Generate a random sparse matrix with specified dimensions and density
fn generate_sparse_matrix(rows: usize, cols: usize, density: f64) -> SparseMatrixCSR<f32> {
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    
    for _ in 0..rows {
        let nnz_in_row = ((cols as f64 * density) as usize).max(1);
        let mut cols_in_row: Vec<usize> = (0..cols).collect();
        
        // Randomly select columns
        use rand::seq::SliceRandom;
        cols_in_row.shuffle(&mut rng);
        cols_in_row.truncate(nnz_in_row);
        cols_in_row.sort_unstable();
        
        for col in cols_in_row {
            col_idx.push(col);
            values.push(rng.gen::<f32>());
        }
        
        row_ptr.push(col_idx.len());
    }
    
    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

/// Measure memory bandwidth utilization
fn measure_memory_bandwidth(size_mb: f64, time_secs: f64) -> f64 {
    size_mb / time_secs // MB/s
}

/// Benchmark baseline SpGEMM without prefetching
fn bench_baseline_no_prefetch(c: &mut Criterion) {
    let mut group = c.benchmark_group("spgemm_no_prefetch");
    
    for &size in &[100, 500, 1000, 2000] {
        let density = 0.05; // 5% density
        let a = generate_sparse_matrix(size, size, density);
        let b = generate_sparse_matrix(size, size, density);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(a.clone(), b.clone()),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let config = MagnusConfig::default();
                    black_box(magnus_spgemm(a, b, &config))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different prefetch strategies
fn bench_prefetch_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_strategies");
    
    // Test matrix: 1000x1000 with 5% density
    let size = 1000;
    let density = 0.05;
    let a = generate_sparse_matrix(size, size, density);
    let b = generate_sparse_matrix(size, size, density);
    
    // Baseline - no prefetch
    group.bench_function("none", |bencher| {
        bencher.iter(|| {
            let config = MagnusConfig::default();
            black_box(magnus_spgemm(&a, &b, &config))
        });
    });
    
    // NOTE: Actual prefetch implementations will be added after we implement them
    // For now, these are placeholders that use the default config
    
    // Conservative - prefetch next row only
    group.bench_function("conservative", |bencher| {
        bencher.iter(|| {
            let config = MagnusConfig::default(); // Will use prefetch_conservative()
            black_box(magnus_spgemm(&a, &b, &config))
        });
    });
    
    // Moderate - prefetch next row + B matrix rows
    group.bench_function("moderate", |bencher| {
        bencher.iter(|| {
            let config = MagnusConfig::default(); // Will use prefetch_moderate()
            black_box(magnus_spgemm(&a, &b, &config))
        });
    });
    
    // Aggressive - full lookahead
    group.bench_function("aggressive", |bencher| {
        bencher.iter(|| {
            let config = MagnusConfig::default(); // Will use prefetch_aggressive()
            black_box(magnus_spgemm(&a, &b, &config))
        });
    });
    
    group.finish();
}

/// Benchmark memory overhead of prefetching
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_memory_overhead");
    group.sample_size(10);
    
    // Large matrix to measure memory impact
    let size = 5000;
    let density = 0.01; // 1% density for large matrix
    let a = generate_sparse_matrix(size, size, density);
    let b = generate_sparse_matrix(size, size, density);
    
    // Measure baseline memory usage
    group.bench_function("baseline_memory", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total_time = std::time::Duration::ZERO;
            
            for _ in 0..iters {
                // Get memory before
                let before = get_memory_usage();
                
                let start = Instant::now();
                let config = MagnusConfig::default();
                let _ = magnus_spgemm(&a, &b, &config);
                total_time += start.elapsed();
                
                // Get memory after
                let after = get_memory_usage();
                black_box(after - before); // Track memory delta
            }
            
            total_time
        });
    });
    
    group.finish();
}

/// Get current memory usage (simplified - in production use proper memory tracking)
fn get_memory_usage() -> usize {
    // This is a placeholder - in production, we'd use:
    // - /proc/self/status on Linux
    // - mach_task_info on macOS
    // - GetProcessMemoryInfo on Windows
    0
}

/// Benchmark cache efficiency with different matrix patterns
fn bench_cache_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_patterns");
    
    // Different sparsity patterns affect cache behavior
    let patterns = [
        ("uniform", 0.05),      // Uniform distribution
        ("clustered", 0.05),    // Clustered non-zeros (better cache)
        ("random", 0.05),       // Random pattern (worst cache)
        ("diagonal", 0.05),     // Diagonal-heavy (sequential access)
    ];
    
    for (pattern_name, density) in patterns {
        let size = 1000;
        let a = generate_pattern_matrix(size, size, density, pattern_name);
        let b = generate_pattern_matrix(size, size, density, pattern_name);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(pattern_name),
            &(a, b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let config = MagnusConfig::default();
                    black_box(magnus_spgemm(a, b, &config))
                });
            },
        );
    }
    
    group.finish();
}

/// Generate matrix with specific sparsity pattern
fn generate_pattern_matrix(rows: usize, cols: usize, density: f64, pattern: &str) -> SparseMatrixCSR<f32> {
    match pattern {
        "clustered" => {
            // Create clusters of non-zeros (cache-friendly)
            generate_clustered_matrix(rows, cols, density)
        },
        "diagonal" => {
            // Diagonal-heavy pattern (sequential access)
            generate_diagonal_matrix(rows, cols, density)
        },
        _ => {
            // Default to uniform random
            generate_sparse_matrix(rows, cols, density)
        }
    }
}

fn generate_clustered_matrix(rows: usize, cols: usize, density: f64) -> SparseMatrixCSR<f32> {
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    
    let cluster_size = 10;
    let nnz_per_row = ((cols as f64 * density) as usize).max(1);
    
    for i in 0..rows {
        // Create clusters around diagonal
        let start_col = (i as i32 - cluster_size / 2).max(0) as usize;
        let end_col = ((i as i32 + cluster_size / 2) as usize).min(cols);
        
        let mut cols_in_row = Vec::new();
        for _ in 0..nnz_per_row {
            if start_col < end_col {
                cols_in_row.push(rng.gen_range(start_col..end_col));
            }
        }
        
        cols_in_row.sort_unstable();
        cols_in_row.dedup();
        
        for col in cols_in_row {
            col_idx.push(col);
            values.push(rng.gen::<f32>());
        }
        
        row_ptr.push(col_idx.len());
    }
    
    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

fn generate_diagonal_matrix(rows: usize, cols: usize, density: f64) -> SparseMatrixCSR<f32> {
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..rows {
        // Always include diagonal element
        if i < cols {
            col_idx.push(i);
            values.push(1.0 + rng.gen::<f32>());
        }
        
        // Add some off-diagonal elements
        let extra_nnz = ((cols as f64 * density) as usize).saturating_sub(1);
        for j in 0..extra_nnz {
            let col = (i + j + 1) % cols;
            col_idx.push(col);
            values.push(rng.gen::<f32>());
        }
        
        row_ptr.push(col_idx.len());
    }
    
    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

criterion_group!(
    benches,
    bench_baseline_no_prefetch,
    bench_prefetch_strategies,
    bench_memory_overhead,
    bench_cache_patterns
);
criterion_main!(benches);