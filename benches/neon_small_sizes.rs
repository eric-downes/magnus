//! Benchmark NEON for small sizes where it should excel

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use magnus::accumulator::{FallbackAccumulator, SimdAccelerator};
use std::hint::black_box;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
use magnus::accumulator::NeonAccumulator;

fn bench_small_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_sizes");

    // Test exact sizes that match NEON implementations
    let sizes = vec![4, 8, 16, 32];

    for size in sizes {
        // Generate test data
        let indices: Vec<usize> = (0..size).map(|i| (i * 3) % (size * 2)).collect();
        let values: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();

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

/// Test with pre-sorted data (accumulation only)
fn bench_accumulation_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulation_only");

    let sizes = vec![8, 16, 32];

    for size in sizes {
        // Generate sorted data with duplicates
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..size {
            indices.push(i / 2); // Create duplicates
            values.push(i as f32);
        }

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

criterion_group!(benches, bench_small_sizes, bench_accumulation_only);
criterion_main!(benches);
