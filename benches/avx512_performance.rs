//! Performance benchmarks for AVX512 accumulator

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(target_arch = "x86_64")]
use magnus::accumulator::avx512::{Avx512Accumulator, is_avx512_available};
use magnus::accumulator::{Accumulator, sort::SortAccumulator};
use std::collections::HashMap;

/// Generate test data with specified duplicate ratio
fn generate_test_data(size: usize, duplicate_ratio: f32) -> (Vec<usize>, Vec<f32>) {
    let unique_count = ((size as f32) * (1.0 - duplicate_ratio)).max(1.0) as usize;
    let mut indices = Vec::with_capacity(size);
    let mut values = Vec::with_capacity(size);
    
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    for i in 0..size {
        let idx = rng.gen_range(0..unique_count);
        let val = rng.gen::<f32>() * 100.0;
        indices.push(idx);
        values.push(val);
    }
    
    (indices, values)
}

fn benchmark_accumulator_variants(c: &mut Criterion) {
    let sizes = vec![16, 32, 64, 128, 256, 512, 1024, 4096];
    let duplicate_ratios = vec![0.0, 0.1, 0.3, 0.5];
    
    let mut group = c.benchmark_group("accumulator_comparison");
    
    for size in &sizes {
        for ratio in &duplicate_ratios {
            let (indices, values) = generate_test_data(*size, *ratio);
            let test_name = format!("size_{}_dup_{:.1}", size, ratio);
            
            // Benchmark generic sort accumulator
            group.bench_with_input(
                BenchmarkId::new("sort_generic", &test_name),
                &(&indices, &values),
                |b, (idx, val)| {
                    b.iter(|| {
                        let mut acc = SortAccumulator::new(*size);
                        for (i, v) in idx.iter().zip(val.iter()) {
                            acc.accumulate(*i, *v);
                        }
                        black_box(acc.extract_result());
                    });
                },
            );
            
            // Benchmark AVX512 accumulator if available
            #[cfg(target_arch = "x86_64")]
            {
                if is_avx512_available() {
                    group.bench_with_input(
                        BenchmarkId::new("avx512", &test_name),
                        &(&indices, &values),
                        |b, (idx, val)| {
                            b.iter(|| {
                                let mut acc = Avx512Accumulator::new(*size);
                                for (i, v) in idx.iter().zip(val.iter()) {
                                    acc.accumulate(*i, *v as f32);
                                }
                                black_box(acc.extract_result());
                            });
                        },
                    );
                }
            }
        }
    }
    
    group.finish();
}

fn benchmark_avx512_sizes(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_avx512_available() {
            eprintln!("Skipping AVX512 benchmarks - not available on this CPU");
            return;
        }
        
        let mut group = c.benchmark_group("avx512_size_scaling");
        
        // Test powers of 2 and multiples of 16 (AVX512 vector size)
        let sizes = vec![
            16,    // 1 vector
            32,    // 2 vectors
            48,    // 3 vectors
            64,    // 4 vectors
            128,   // 8 vectors
            256,   // 16 vectors
            512,   // 32 vectors
            1024,  // 64 vectors
        ];
        
        for size in sizes {
            let (indices, values) = generate_test_data(size, 0.2);
            
            group.bench_function(BenchmarkId::new("avx512", size), |b| {
                b.iter(|| {
                    let mut acc = Avx512Accumulator::new(size);
                    for (i, v) in indices.iter().zip(values.iter()) {
                        acc.accumulate(*i, *v);
                    }
                    black_box(acc.extract_result());
                });
            });
        }
        
        group.finish();
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!("AVX512 benchmarks not available on this architecture");
    }
}

fn benchmark_duplicate_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("duplicate_accumulation");
    
    let size = 256;
    let duplicate_ratios = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    
    for ratio in duplicate_ratios {
        let (indices, values) = generate_test_data(size, ratio);
        
        // Benchmark sort accumulator
        group.bench_function(
            BenchmarkId::new("sort", format!("dup_{:.1}", ratio)),
            |b| {
                b.iter(|| {
                    let mut acc = SortAccumulator::new(size);
                    for (i, v) in indices.iter().zip(values.iter()) {
                        acc.accumulate(*i, *v);
                    }
                    black_box(acc.extract_result());
                });
            },
        );
        
        // Benchmark AVX512 if available
        #[cfg(target_arch = "x86_64")]
        {
            if is_avx512_available() {
                group.bench_function(
                    BenchmarkId::new("avx512", format!("dup_{:.1}", ratio)),
                    |b| {
                        b.iter(|| {
                            let mut acc = Avx512Accumulator::new(size);
                            for (i, v) in indices.iter().zip(values.iter()) {
                                acc.accumulate(*i, *v);
                            }
                            black_box(acc.extract_result());
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

fn benchmark_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case");
    
    // Worst case: all unique indices in reverse order
    let size = 1024;
    let indices: Vec<usize> = (0..size).rev().collect();
    let values: Vec<f32> = (0..size).map(|i| i as f32).collect();
    
    group.bench_function("sort_worst", |b| {
        b.iter(|| {
            let mut acc = SortAccumulator::new(size);
            for (i, v) in indices.iter().zip(values.iter()) {
                acc.accumulate(*i, *v);
            }
            black_box(acc.extract_result());
        });
    });
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_available() {
            group.bench_function("avx512_worst", |b| {
                b.iter(|| {
                    let mut acc = Avx512Accumulator::new(size);
                    for (i, v) in indices.iter().zip(values.iter()) {
                        acc.accumulate(*i, *v);
                    }
                    black_box(acc.extract_result());
                });
            });
        }
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_accumulator_variants,
    benchmark_avx512_sizes,
    benchmark_duplicate_handling,
    benchmark_worst_case
);
criterion_main!(benches);