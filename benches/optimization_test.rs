//! Quick benchmark to test the new optimizations

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use magnus::accumulator::{NeonAccumulator, AccelerateAccumulator, FallbackAccumulator, SimdAccelerator};
use rand::Rng;

fn bench_accumulator_32_elements(c: &mut Criterion) {
    let mut group = c.benchmark_group("32_elements");
    
    // Generate test data
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..32).map(|i| i % 20).collect();
    let values: Vec<f32> = (0..32).map(|_| rng.gen::<f32>()).collect();
    
    // Shuffle indices
    use rand::seq::SliceRandom;
    indices.shuffle(&mut rng);
    
    group.bench_function("Fallback", |b| {
        let acc = FallbackAccumulator::new();
        b.iter(|| {
            black_box(acc.sort_and_accumulate(&indices, &values))
        });
    });
    
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        group.bench_function("NEON_Optimized", |b| {
            let acc = NeonAccumulator::new();
            b.iter(|| {
                black_box(acc.sort_and_accumulate(&indices, &values))
            });
        });
        
        group.bench_function("Accelerate", |b| {
            let acc = AccelerateAccumulator::new();
            b.iter(|| {
                black_box(acc.sort_and_accumulate(&indices, &values))
            });
        });
    }
    
    group.finish();
}

fn bench_accumulator_64_elements(c: &mut Criterion) {
    let mut group = c.benchmark_group("64_elements");
    
    // Generate test data
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..64).map(|i| i % 40).collect();
    let values: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
    
    // Shuffle indices
    use rand::seq::SliceRandom;
    indices.shuffle(&mut rng);
    
    group.bench_function("Fallback", |b| {
        let acc = FallbackAccumulator::new();
        b.iter(|| {
            black_box(acc.sort_and_accumulate(&indices, &values))
        });
    });
    
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        group.bench_function("NEON_Hybrid", |b| {
            let acc = NeonAccumulator::new();
            b.iter(|| {
                black_box(acc.sort_and_accumulate(&indices, &values))
            });
        });
        
        group.bench_function("Accelerate", |b| {
            let acc = AccelerateAccumulator::new();
            b.iter(|| {
                black_box(acc.sort_and_accumulate(&indices, &values))
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_accumulator_32_elements, bench_accumulator_64_elements);
criterion_main!(benches);