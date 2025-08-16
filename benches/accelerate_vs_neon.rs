//! Benchmark comparing Accelerate framework vs pure NEON implementation

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use magnus::accumulator::{NeonAccumulator, AccelerateAccumulator, SimdAccelerator};
use rand::Rng;
use rand::seq::SliceRandom;

fn generate_test_data(size: usize, num_unique: usize) -> (Vec<usize>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..size).map(|i| i % num_unique).collect();
    let values: Vec<f32> = (0..size).map(|_| rng.gen::<f32>()).collect();
    indices.shuffle(&mut rng);
    (indices, values)
}

fn bench_comparison(c: &mut Criterion) {
    // Test different sizes to see where each implementation excels
    let test_cases = [
        (4, 4, "size_4"),
        (8, 6, "size_8"), 
        (16, 12, "size_16"),
        (32, 20, "size_32"),
        (64, 40, "size_64"),
        (128, 80, "size_128"),
        (256, 150, "size_256"),
        (512, 300, "size_512"),
        (1024, 600, "size_1024"),
    ];
    
    for (size, num_unique, name) in test_cases {
        let mut group = c.benchmark_group(format!("accumulator_{}", name));
        let (indices, values) = generate_test_data(size, num_unique);
        
        // Benchmark NEON
        group.bench_function("NEON", |b| {
            let acc = NeonAccumulator::new();
            b.iter(|| {
                black_box(acc.sort_and_accumulate(&indices, &values))
            });
        });
        
        // Benchmark Accelerate
        group.bench_function("Accelerate", |b| {
            let acc = AccelerateAccumulator::new();
            b.iter(|| {
                black_box(acc.sort_and_accumulate(&indices, &values))
            });
        });
        
        group.finish();
    }
}

fn bench_large_sizes(c: &mut Criterion) {
    // Test larger sizes where Accelerate should theoretically excel
    let test_cases = [
        (2048, 1200, "size_2k"),
        (4096, 2400, "size_4k"),
        (8192, 4800, "size_8k"),
    ];
    
    for (size, num_unique, name) in test_cases {
        let mut group = c.benchmark_group(format!("large_{}", name));
        group.sample_size(10); // Fewer samples for large sizes
        
        let (indices, values) = generate_test_data(size, num_unique);
        
        // Benchmark NEON
        group.bench_function("NEON", |b| {
            let acc = NeonAccumulator::new();
            b.iter(|| {
                black_box(acc.sort_and_accumulate(&indices, &values))
            });
        });
        
        // Benchmark Accelerate
        group.bench_function("Accelerate", |b| {
            let acc = AccelerateAccumulator::new();
            b.iter(|| {
                black_box(acc.sort_and_accumulate(&indices, &values))
            });
        });
        
        group.finish();
    }
}

// Also test the edge cases where we transition between implementations
fn bench_transition_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition_points");
    
    // Test around the 32-element boundary (where Accelerate switches from NEON)
    for size in [30, 31, 32, 33, 34, 35] {
        let (indices, values) = generate_test_data(size, size * 3 / 4);
        
        group.bench_with_input(
            BenchmarkId::new("NEON", size),
            &(indices.clone(), values.clone()),
            |b, (idx, val)| {
                let acc = NeonAccumulator::new();
                b.iter(|| black_box(acc.sort_and_accumulate(idx, val)));
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("Accelerate", size),
            &(indices, values),
            |b, (idx, val)| {
                let acc = AccelerateAccumulator::new();
                b.iter(|| black_box(acc.sort_and_accumulate(idx, val)));
            }
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_comparison, bench_large_sizes, bench_transition_sizes);
criterion_main!(benches);