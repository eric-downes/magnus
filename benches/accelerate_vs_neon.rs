//! Benchmark comparing Accelerate framework vs pure NEON implementation

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod apple_benchmarks {
    use criterion::Criterion;
    use magnus::accumulator::{AccelerateAccumulator, NeonAccumulator, SimdAccelerator};
    use rand::seq::SliceRandom;
    use rand::Rng;
    use std::hint::black_box;

    fn generate_test_data(size: usize, num_unique: usize) -> (Vec<usize>, Vec<f32>) {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..size).map(|i| i % num_unique).collect();
        indices.shuffle(&mut rng);

        let values: Vec<f32> = (0..size).map(|_| rng.gen_range(0.1..10.0)).collect();
        (indices, values)
    }

    pub fn bench_accumulator_comparison(c: &mut Criterion) {
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
                b.iter(|| black_box(acc.sort_and_accumulate(&indices, &values)));
            });

            // Benchmark Accelerate
            group.bench_function("Accelerate", |b| {
                let acc = AccelerateAccumulator::new();
                b.iter(|| black_box(acc.sort_and_accumulate(&indices, &values)));
            });

            group.finish();
        }
    }

    pub fn bench_large_accumulator(c: &mut Criterion) {
        // Test larger sizes where Accelerate should dominate
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
                b.iter(|| black_box(acc.sort_and_accumulate(&indices, &values)));
            });

            // Benchmark Accelerate
            group.bench_function("Accelerate", |b| {
                let acc = AccelerateAccumulator::new();
                b.iter(|| black_box(acc.sort_and_accumulate(&indices, &values)));
            });

            group.finish();
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn bench_accumulator_comparison(c: &mut Criterion) {
    apple_benchmarks::bench_accumulator_comparison(c);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn bench_large_accumulator(c: &mut Criterion) {
    apple_benchmarks::bench_large_accumulator(c);
}

// Provide stub benchmarks for non-Apple platforms
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn bench_accumulator_comparison(_c: &mut Criterion) {
    // No-op on non-Apple platforms
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn bench_large_accumulator(_c: &mut Criterion) {
    // No-op on non-Apple platforms
}

criterion_group!(
    benches,
    bench_accumulator_comparison,
    bench_large_accumulator
);
criterion_main!(benches);
