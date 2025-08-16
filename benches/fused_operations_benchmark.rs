//! Benchmark to verify fused operations provide >15% improvement for high-duplicate cases

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use magnus::accumulator::{
    AccelerateAccumulator, SimdAccelerator,
    fused::fused_sort_accumulate,
    duplicate_prediction::DuplicateContext,
    adaptive_sort::AdaptiveSortAccumulator,
    Accumulator,
};
use rand::Rng;

/// Generate test data with controlled duplicate rate
fn generate_high_duplicate_data(
    n_elements: usize,
    n_unique: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut indices = Vec::with_capacity(n_elements);
    let mut values = Vec::with_capacity(n_elements);
    
    for _ in 0..n_elements {
        indices.push(rng.gen_range(0..n_unique));
        values.push(rng.gen::<f32>());
    }
    
    (indices, values)
}

/// Benchmark separated approach (Accelerate sort + accumulate)
fn bench_separated(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_duplicate_75_percent");
    
    // Test case: 75% duplicates (4x products to columns)
    let n_elements = 10000;
    let n_unique = 2500; // 75% duplicates
    let (indices, values) = generate_high_duplicate_data(n_elements, n_unique);
    
    group.bench_function("separated_accelerate", |b| {
        b.iter(|| {
            let indices_clone = indices.clone();
            let values_clone = values.clone();
            
            // Use Accelerate for sorting
            let acc = AccelerateAccumulator::new();
            let result = acc.sort_and_accumulate(&indices_clone, &values_clone);
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark fused approach
fn bench_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_duplicate_75_percent");
    
    // Same test case: 75% duplicates
    let n_elements = 10000;
    let n_unique = 2500;
    let (indices, values) = generate_high_duplicate_data(n_elements, n_unique);
    
    group.bench_function("fused_sort_accumulate", |b| {
        b.iter(|| {
            let indices_clone = indices.clone();
            let values_clone = values.clone();
            
            let result = fused_sort_accumulate(indices_clone, values_clone);
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark adaptive approach (should choose fused for high duplicates)
fn bench_adaptive(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_duplicate_75_percent");
    
    // Same test case: 75% duplicates
    let n_elements = 10000;
    let n_unique = 2500;
    let (indices, values) = generate_high_duplicate_data(n_elements, n_unique);
    
    group.bench_function("adaptive", |b| {
        b.iter(|| {
            let indices_clone = indices.clone();
            let values_clone = values.clone();
            
            // Create context that predicts high duplicates
            let context = DuplicateContext {
                b_ncols: n_unique,
                expected_products: n_elements,
            };
            
            let mut acc = AdaptiveSortAccumulator::with_context(256, context);
            
            // Accumulate all elements
            for (idx, val) in indices_clone.into_iter().zip(values_clone.into_iter()) {
                acc.accumulate(idx, val);
            }
            
            let result = acc.extract_result();
            black_box(result)
        });
    });
    
    group.finish();
}

/// Compare performance across different duplicate rates
fn bench_duplicate_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("duplicate_spectrum");
    
    let n_elements = 5000;
    
    for duplicate_rate in [0.0, 0.25, 0.50, 0.75, 0.90] {
        let n_unique = ((1.0 - duplicate_rate) * n_elements as f64) as usize;
        let n_unique = n_unique.max(1);
        
        let (indices, values) = generate_high_duplicate_data(n_elements, n_unique);
        
        group.bench_with_input(
            BenchmarkId::new("separated", format!("{:.0}%", duplicate_rate * 100.0)),
            &(indices.clone(), values.clone()),
            |b, (idx, val)| {
                b.iter(|| {
                    let acc = AccelerateAccumulator::new();
                    acc.sort_and_accumulate(idx, val)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("fused", format!("{:.0}%", duplicate_rate * 100.0)),
            &(indices.clone(), values.clone()),
            |b, (idx, val)| {
                b.iter(|| {
                    fused_sort_accumulate(idx.clone(), val.clone())
                });
            },
        );
    }
    
    group.finish();
}

/// Verify correctness (not a benchmark, but important)
fn verify_correctness() {
    let n_elements = 1000;
    let n_unique = 100;
    let (indices, values) = generate_high_duplicate_data(n_elements, n_unique);
    
    // Get results from both approaches
    let acc = AccelerateAccumulator::new();
    let (sep_idx, sep_val) = acc.sort_and_accumulate(&indices, &values);
    
    let (fused_idx, fused_val) = fused_sort_accumulate(indices.clone(), values.clone());
    
    // Results should be identical
    assert_eq!(sep_idx.len(), fused_idx.len(), "Different number of unique indices");
    
    for i in 0..sep_idx.len() {
        assert_eq!(sep_idx[i], fused_idx[i], "Index mismatch at position {}", i);
        
        // Allow small floating point difference
        let diff = (sep_val[i] - fused_val[i]).abs();
        assert!(diff < 1e-5, "Value mismatch at position {}: {} vs {}", i, sep_val[i], fused_val[i]);
    }
    
    println!("✓ Correctness verified: Both approaches produce identical results");
}

/// Main benchmark entry point
fn run_verification(c: &mut Criterion) {
    // First verify correctness
    verify_correctness();
    
    // Then run benchmarks
    bench_separated(c);
    bench_fused(c);
    bench_adaptive(c);
    bench_duplicate_spectrum(c);
}

criterion_group!(benches, run_verification);
criterion_main!(benches);