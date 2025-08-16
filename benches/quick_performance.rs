use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use magnus::{
    accumulator::{FallbackAccumulator, SimdAccelerator},
    detect_architecture, magnus_spgemm, Architecture, MagnusConfig, SparseMatrixCSR,
};

/// Generate a simple sparse matrix for benchmarking
fn generate_sparse_matrix(n: usize, density: f64) -> SparseMatrixCSR<f32> {
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        let nnz = ((n as f64) * density) as usize;
        for j in 0..nnz.min(n) {
            let col = (j * 3 + i * 7) % n;
            col_idx.push(col);
            values.push(1.0 + (i * j) as f32 * 0.1);
        }
        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

/// Quick benchmark comparing NEON vs Fallback
fn bench_quick_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_comparison");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));
    
    println!("\n=== Performance Test on Apple Silicon ===");
    println!("Detected architecture: {:?}", detect_architecture());
    
    // Test data
    let sizes = vec![256, 512, 1024];
    let col_indices: Vec<usize> = (0..256).map(|i| i % 64).collect();
    let values: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
    
    // Benchmark NEON if available
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        use magnus::accumulator::NeonAccumulator;
        let neon = NeonAccumulator::new();
        
        for &size in &sizes {
            let test_cols = &col_indices[..size.min(256)];
            let test_vals = &values[..size.min(256)];
            
            group.bench_with_input(
                BenchmarkId::new("NEON", size),
                &(test_cols, test_vals),
                |b, (cols, vals)| {
                    b.iter(|| {
                        let (c, v) = neon.sort_and_accumulate(cols, vals);
                        black_box((c, v))
                    })
                },
            );
        }
    }
    
    // Benchmark Fallback for comparison
    let fallback = FallbackAccumulator::new();
    for &size in &sizes {
        let test_cols = &col_indices[..size.min(256)];
        let test_vals = &values[..size.min(256)];
        
        group.bench_with_input(
            BenchmarkId::new("Fallback", size),
            &(test_cols, test_vals),
            |b, (cols, vals)| {
                b.iter(|| {
                    let (c, v) = fallback.sort_and_accumulate(cols, vals);
                    black_box((c, v))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark different thresholds
fn bench_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_comparison");
    group.sample_size(10);
    
    let thresholds = vec![128, 192, 256];
    let a = generate_sparse_matrix(50, 0.1);
    let b = generate_sparse_matrix(50, 0.1);
    
    for threshold in thresholds {
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

/// Compare ARM-optimized vs Generic
fn bench_arm_vs_generic(c: &mut Criterion) {
    let mut group = c.benchmark_group("arm_vs_generic");
    group.sample_size(10);
    
    println!("\n=== ARM-Optimized vs Generic Performance ===");
    
    let sizes = vec![(50, 0.1), (100, 0.05)];
    
    for (size, density) in sizes {
        let a = generate_sparse_matrix(size, density);
        let b = generate_sparse_matrix(size, density);
        
        // ARM-optimized
        let arm_config = MagnusConfig::for_architecture(Architecture::ArmNeon);
        group.bench_with_input(
            BenchmarkId::new("ARM", size),
            &(&a, &b, &arm_config),
            |bench, (a, b, config)| {
                bench.iter(|| {
                    let c = magnus_spgemm(a, b, config);
                    black_box(c)
                })
            },
        );
        
        // Generic
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
    
    // Print summary
    println!("\nExpected improvements from ARM optimization:");
    println!("- Reduced accumulator threshold (192 vs 256)");
    println!("- Optimized chunk size (512 vs 256)");
    println!("- NEON SIMD operations where available");
}

criterion_group!(
    benches,
    bench_quick_comparison,
    bench_thresholds,
    bench_arm_vs_generic
);
criterion_main!(benches);