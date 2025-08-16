//! Diagnostic benchmarks for NEON implementation
//!
//! This helps identify exactly where the performance issues are

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use magnus::accumulator::{FallbackAccumulator, SimdAccelerator};
use std::hint::black_box;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
use magnus::accumulator::NeonAccumulator;

/// Generate test data with known characteristics
fn generate_test_vectors(size: usize) -> (Vec<usize>, Vec<f32>) {
    let mut indices = Vec::with_capacity(size);
    let mut values = Vec::with_capacity(size);

    for i in 0..size {
        // Create data with ~25% duplicates
        // Avoid division by zero for small sizes
        let modulo = (size * 3 / 4).max(1);
        indices.push((i * 3) % modulo);
        values.push(i as f32 * 0.1);
    }

    (indices, values)
}

/// Benchmark the overhead of function calls
fn bench_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_overhead");

    // Test tiny inputs to measure overhead
    for size in [1, 2, 4, 8].iter() {
        let (indices, values) = generate_test_vectors(*size);

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            let neon = NeonAccumulator::new();
            group.bench_with_input(
                BenchmarkId::new("NEON", size),
                &(&indices, &values),
                |b, (idx, val)| {
                    b.iter(|| {
                        let result = neon.sort_and_accumulate(idx, val);
                        black_box(result)
                    })
                },
            );
        }

        let fallback = FallbackAccumulator::new();
        group.bench_with_input(
            BenchmarkId::new("Fallback", size),
            &(&indices, &values),
            |b, (idx, val)| {
                b.iter(|| {
                    let result = fallback.sort_and_accumulate(idx, val);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark specific size categories
fn bench_size_categories(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_categories");

    // Powers of 2 (optimal for bitonic)
    let sizes = vec![4, 8, 16, 32, 64, 128, 256];

    for size in sizes {
        let (indices, values) = generate_test_vectors(size);

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            let neon = NeonAccumulator::new();
            group.bench_with_input(
                BenchmarkId::new("NEON", size),
                &(&indices, &values),
                |b, (idx, val)| {
                    b.iter(|| {
                        let result = neon.sort_and_accumulate(idx, val);
                        black_box(result)
                    })
                },
            );
        }

        let fallback = FallbackAccumulator::new();
        group.bench_with_input(
            BenchmarkId::new("Fallback", size),
            &(&indices, &values),
            |b, (idx, val)| {
                b.iter(|| {
                    let result = fallback.sort_and_accumulate(idx, val);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark accumulation phase separately
fn bench_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulation_only");

    // Pre-sorted data with varying duplicate rates
    let sizes = vec![(100, 10), (100, 25), (100, 50)]; // (size, % duplicates)

    for (size, dup_rate) in sizes {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Generate sorted data with controlled duplicates
        let unique_count = size * (100 - dup_rate) / 100;
        for i in 0..size {
            indices.push(i * unique_count / size);
            values.push(i as f32 * 0.1);
        }

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            let neon = NeonAccumulator::new();
            group.bench_with_input(
                BenchmarkId::new("NEON", format!("size{}_dup{}", size, dup_rate)),
                &(&indices, &values),
                |b, (idx, val)| {
                    b.iter(|| {
                        // Data is already sorted, so this mainly tests accumulation
                        let result = neon.sort_and_accumulate(idx, val);
                        black_box(result)
                    })
                },
            );
        }

        let fallback = FallbackAccumulator::new();
        group.bench_with_input(
            BenchmarkId::new("Fallback", format!("size{}_dup{}", size, dup_rate)),
            &(&indices, &values),
            |b, (idx, val)| {
                b.iter(|| {
                    let result = fallback.sort_and_accumulate(idx, val);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Analyze performance by input characteristics
fn bench_input_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("input_patterns");

    let size = 64;

    // Different input patterns
    let patterns = vec![
        ("random", generate_random_pattern(size)),
        ("sorted", generate_sorted_pattern(size)),
        ("reverse", generate_reverse_pattern(size)),
        ("clustered", generate_clustered_pattern(size)),
    ];

    for (name, (indices, values)) in patterns {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            let neon = NeonAccumulator::new();
            group.bench_with_input(
                BenchmarkId::new("NEON", name),
                &(&indices, &values),
                |b, (idx, val)| {
                    b.iter(|| {
                        let result = neon.sort_and_accumulate(idx, val);
                        black_box(result)
                    })
                },
            );
        }

        let fallback = FallbackAccumulator::new();
        group.bench_with_input(
            BenchmarkId::new("Fallback", name),
            &(&indices, &values),
            |b, (idx, val)| {
                b.iter(|| {
                    let result = fallback.sort_and_accumulate(idx, val);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn generate_random_pattern(size: usize) -> (Vec<usize>, Vec<f32>) {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();

    let indices: Vec<usize> = (0..size).map(|_| rng.gen_range(0..size * 2)).collect();
    let values: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
    (indices, values)
}

fn generate_sorted_pattern(size: usize) -> (Vec<usize>, Vec<f32>) {
    let indices: Vec<usize> = (0..size).map(|i| i * 2).collect();
    let values: Vec<f32> = (0..size).map(|i| i as f32).collect();
    (indices, values)
}

fn generate_reverse_pattern(size: usize) -> (Vec<usize>, Vec<f32>) {
    let indices: Vec<usize> = (0..size).map(|i| (size - i - 1) * 2).collect();
    let values: Vec<f32> = (0..size).map(|i| i as f32).collect();
    (indices, values)
}

fn generate_clustered_pattern(size: usize) -> (Vec<usize>, Vec<f32>) {
    let mut indices = Vec::new();
    let mut values = Vec::new();

    // Create clusters of similar values
    let cluster_size = 8;
    for cluster in 0..(size / cluster_size) {
        for i in 0..cluster_size {
            indices.push(cluster * 10 + i / 2);
            values.push((cluster * cluster_size + i) as f32);
        }
    }

    (indices, values)
}

/// Print diagnostic information
#[allow(dead_code)]
fn print_diagnostics() {
    println!("\n=== NEON Diagnostic Information ===");

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        println!("✅ Running on ARM64 with NEON support");
        println!("Vector width: 128 bits (4 x f32 or 4 x u32)");

        // Test if NEON instructions are available
        unsafe {
            use std::arch::aarch64::*;
            let test = vdupq_n_f32(1.0);
            let sum = vaddvq_f32(test);
            println!("NEON instruction test: {} (expected: 4.0)", sum);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        println!("❌ Not running on ARM64/macOS - NEON not available");
    }

    println!("\nExpected issues to diagnose:");
    println!("1. Function call overhead for small inputs");
    println!("2. Scalar fallback in supposedly NEON code");
    println!("3. Data structure conversion overhead");
    println!("4. Poor memory access patterns");
    println!("");
}

criterion_group!(
    benches,
    bench_overhead,
    bench_size_categories,
    bench_accumulation,
    bench_input_patterns
);

criterion_main!(benches);
