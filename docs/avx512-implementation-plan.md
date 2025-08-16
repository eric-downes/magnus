# AVX512 Optimization Implementation Plan

## Executive Summary

This document outlines a comprehensive plan for implementing AVX512-specific optimizations for the MAGNUS sparse matrix multiplication algorithm. The optimizations focus on the sort-based accumulator component, which is critical for performance when handling sparse intermediate products.

### Phases

0. Architecture Design: Module structure for AVX512 components
1. TDD Phase: Write Tests for all critical and high-level functionality for 2-5
2. Sort-then-reduce implementation using AVX512 intrinsics
3. Modified compare-exchange with in-place accumulation
4. Integration with existing framework and adaptive strategy selection
5. Comprehensive testing and benchmarking suite (weeks 10-12)

## Current State Analysis

### Completed Components
- **Core Algorithm**: Full MAGNUS implementation with row categorization, fine/coarse reordering
- **Accumulator Framework**: Trait-based design supporting dense and sort accumulators
- **Test Infrastructure**: Comprehensive test suite validating correctness
- **Parallel Execution**: Rayon-based parallelization for row-level processing

### Gap Analysis
The primary gap is in hardware-specific optimization for the sort accumulator, specifically:
1. AVX512 vectorized sorting for column indices
2. Accumulation during or after sorting for duplicate column indices
3. Integration with existing accumulator trait interface

## Implementation Notes

### Module Structure

```
src/
├── accumulator/
│   ├── mod.rs              # Trait definition and factory
│   ├── dense.rs            # Dense accumulator
│   ├── sort.rs             # Generic sort accumulator
│   └── avx512/             # NEW: AVX512 optimizations
│       ├── mod.rs          # AVX512 feature detection and dispatch
│       ├── sort_reduce.rs  # Sort-then-reduce implementation
│       ├── bitonic.rs      # Modified compare-exchange
│       └── intrinsics.rs   # Low-level AVX512 operations
```

### Feature Detection and Runtime Dispatch

```rust
// src/accumulator/avx512/mod.rs

use std::arch::x86_64::*;

pub struct Avx512Features {
    pub has_avx512f: bool,   // Foundation
    pub has_avx512dq: bool,  // Double/Quad operations
    pub has_avx512cd: bool,  // Conflict Detection
    pub has_avx512bw: bool,  // Byte/Word operations
}

impl Avx512Features {
    pub fn detect() -> Self {
        if is_x86_feature_detected!("avx512f") {
            Self {
                has_avx512f: true,
                has_avx512dq: is_x86_feature_detected!("avx512dq"),
                has_avx512cd: is_x86_feature_detected!("avx512cd"),
                has_avx512bw: is_x86_feature_detected!("avx512bw"),
            }
        } else {
            Self {
                has_avx512f: false,
                has_avx512dq: false,
                has_avx512cd: false,
                has_avx512bw: false,
            }
        }
    }
}

// Factory function with runtime dispatch
pub fn create_avx512_accumulator<T>(
    initial_capacity: usize,
    features: &Avx512Features,
) -> Box<dyn Accumulator<T>> 
where
    T: Copy + Num + AddAssign + 'static,
{
    if features.has_avx512f && features.has_avx512cd {
        Box::new(Avx512SortAccumulator::new(initial_capacity))
    } else {
        Box::new(sort::SortAccumulator::new(initial_capacity))
    }
}
```

## 1. TDD Phase

TBD

## 2. Sort Phase: AVX512 Sort-Then-Reduce Implementation

### Objective
Implement a vectorized sort-then-reduce accumulator using AVX512 instructions for improved performance on Intel processors.

### 2.1 Data Layout and Alignment

```rust
// src/accumulator/avx512/sort_reduce.rs

use aligned_vec::{AVec, avec};
use std::arch::x86_64::*;

pub struct Avx512SortAccumulator<T> {
    // 64-byte aligned for AVX512 (zmm registers)
    col_indices: AVec<u32, 64>,  // Use u32 for column indices
    values: AVec<T, 64>,
    capacity: usize,
    size: usize,
}

impl<T> Avx512SortAccumulator<T> {
    pub fn new(initial_capacity: usize) -> Self {
        // Round up to multiple of 16 (AVX512 processes 16 elements)
        let aligned_capacity = ((initial_capacity + 15) / 16) * 16;
        
        Self {
            col_indices: avec![0u32; aligned_capacity],
            values: avec![T::zero(); aligned_capacity],
            capacity: aligned_capacity,
            size: 0,
        }
    }
}
```

### 2.2 AVX512 Sorting Network

```rust
// src/accumulator/avx512/intrinsics.rs

// Key-value pair sorting using AVX512
#[target_feature(enable = "avx512f")]
unsafe fn avx512_sort_16_pairs(
    indices: &mut [u32; 16],
    values: &mut [f32; 16],
) {
    // Load 16 indices and values into zmm registers
    let idx_vec = _mm512_loadu_si512(indices.as_ptr() as *const i32);
    let val_vec = _mm512_loadu_ps(values.as_ptr());
    
    // Bitonic sorting network for 16 elements
    // Stage 1: Compare and swap adjacent pairs
    let mask1 = 0xAAAA; // 1010101010101010
    let idx_perm1 = _mm512_set_epi32(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
    
    let idx_shuffled = _mm512_permutexvar_epi32(idx_perm1, idx_vec);
    let val_shuffled = _mm512_permutexvar_ps(idx_perm1, val_vec);
    
    let cmp_mask = _mm512_cmp_epi32_mask(idx_vec, idx_shuffled, _MM_CMPINT_GT);
    
    // Blend based on comparison
    let idx_min = _mm512_mask_blend_epi32(cmp_mask, idx_vec, idx_shuffled);
    let idx_max = _mm512_mask_blend_epi32(cmp_mask, idx_shuffled, idx_vec);
    let val_min = _mm512_mask_blend_ps(cmp_mask, val_vec, val_shuffled);
    let val_max = _mm512_mask_blend_ps(cmp_mask, val_shuffled, val_vec);
    
    // Continue with remaining stages...
    // (Full implementation would include all log2(16) = 4 stages)
    
    // Store sorted results
    _mm512_storeu_si512(indices.as_mut_ptr() as *mut i32, idx_final);
    _mm512_storeu_ps(values.as_mut_ptr(), val_final);
}
```

### 2.3 Vectorized Merge and Accumulation

```rust
// src/accumulator/avx512/sort_reduce.rs

#[target_feature(enable = "avx512f,avx512cd")]
unsafe fn avx512_merge_duplicates(
    sorted_indices: &[u32],
    sorted_values: &[f32],
    out_indices: &mut Vec<u32>,
    out_values: &mut Vec<f32>,
) {
    let mut i = 0;
    let len = sorted_indices.len();
    
    // Process 16 elements at a time
    while i + 16 <= len {
        let idx_vec = _mm512_loadu_si512(&sorted_indices[i] as *const u32 as *const i32);
        let val_vec = _mm512_loadu_ps(&sorted_values[i]);
        
        // Detect conflicts (duplicates) using AVX512CD
        let conflict_mask = _mm512_conflict_epi32(idx_vec);
        
        if conflict_mask == 0 {
            // No duplicates in this batch
            for j in 0..16 {
                out_indices.push(sorted_indices[i + j]);
                out_values.push(sorted_values[i + j]);
            }
        } else {
            // Handle duplicates with accumulation
            let mut accumulated = vec![0.0f32; 16];
            let mut mask = vec![false; 16];
            
            for j in 0..16 {
                if !mask[j] {
                    let current_idx = sorted_indices[i + j];
                    accumulated[j] = sorted_values[i + j];
                    
                    // Find and accumulate duplicates
                    for k in (j + 1)..16 {
                        if sorted_indices[i + k] == current_idx {
                            accumulated[j] += sorted_values[i + k];
                            mask[k] = true;
                        }
                    }
                    
                    out_indices.push(current_idx);
                    out_values.push(accumulated[j]);
                }
            }
        }
        
        i += 16;
    }
    
    // Handle remaining elements with scalar code
    while i < len {
        let mut current_idx = sorted_indices[i];
        let mut current_val = sorted_values[i];
        i += 1;
        
        while i < len && sorted_indices[i] == current_idx {
            current_val += sorted_values[i];
            i += 1;
        }
        
        out_indices.push(current_idx);
        out_values.push(current_val);
    }
}
```

### 2.4: Does it work?

Debug etc

## Phase 2: Modified Compare-Exchange Implementation (Weeks 4-6)

### Objective
Implement a bitonic sorting network that performs accumulation during compare-exchange operations, potentially reducing memory bandwidth requirements.

### 3.1 Modified Compare-Exchange Network

```rust
// src/accumulator/avx512/bitonic.rs

#[derive(Clone, Copy)]
struct IndexValuePair {
    index: u32,
    value: f32,
}

#[target_feature(enable = "avx512f")]
unsafe fn modified_compare_exchange(
    a: IndexValuePair,
    b: IndexValuePair,
) -> (IndexValuePair, IndexValuePair) {
    if a.index < b.index {
        (a, b)
    } else if a.index > b.index {
        (b, a)
    } else {
        // Equal indices: accumulate values
        (
            IndexValuePair {
                index: a.index,
                value: a.value + b.value,
            },
            IndexValuePair {
                index: u32::MAX,  // Sentinel for deletion
                value: 0.0,
            },
        )
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn avx512_bitonic_sort_with_accumulation(
    pairs: &mut [IndexValuePair],
) {
    let n = pairs.len();
    
    // Bitonic sort with modified compare-exchange
    let mut k = 2;
    while k <= n {
        let mut j = k / 2;
        while j > 0 {
            for i in 0..n {
                let l = i ^ j;
                if l > i {
                    if (i & k) == 0 && pairs[i].index > pairs[l].index ||
                       (i & k) != 0 && pairs[i].index < pairs[l].index {
                        pairs.swap(i, l);
                    } else if pairs[i].index == pairs[l].index {
                        // Accumulate on equality
                        pairs[i].value += pairs[l].value;
                        pairs[l].index = u32::MAX;  // Mark for removal
                    }
                }
            }
            j /= 2;
        }
        k *= 2;
    }
    
    // Compact array to remove sentinels
    let mut write_pos = 0;
    for read_pos in 0..n {
        if pairs[read_pos].index != u32::MAX {
            if write_pos != read_pos {
                pairs[write_pos] = pairs[read_pos];
            }
            write_pos += 1;
        }
    }
    pairs.truncate(write_pos);
}
```

### 3.2 Vectorized Compare-Exchange Operations

```rust
// Vectorized version processing 16 pairs at once
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn avx512_compare_exchange_16(
    idx_a: __m512i,
    val_a: __m512,
    idx_b: __m512i,
    val_b: __m512,
) -> (__m512i, __m512, __m512i, __m512) {
    // Compare indices
    let lt_mask = _mm512_cmp_epi32_mask(idx_a, idx_b, _MM_CMPINT_LT);
    let eq_mask = _mm512_cmp_epi32_mask(idx_a, idx_b, _MM_CMPINT_EQ);
    let gt_mask = _mm512_cmp_epi32_mask(idx_a, idx_b, _MM_CMPINT_GT);
    
    // For equal indices, accumulate values
    let sum_val = _mm512_add_ps(val_a, val_b);
    
    // Create output based on comparison
    let out_idx_1 = _mm512_mask_blend_epi32(lt_mask | eq_mask, idx_b, idx_a);
    let out_val_1 = _mm512_mask_blend_ps(eq_mask, 
        _mm512_mask_blend_ps(lt_mask, val_b, val_a), 
        sum_val);
    
    // Second output (sentinel for equal case)
    let sentinel = _mm512_set1_epi32(0xFFFFFFFF);
    let out_idx_2 = _mm512_mask_blend_epi32(eq_mask,
        _mm512_mask_blend_epi32(lt_mask, idx_a, idx_b),
        sentinel);
    let out_val_2 = _mm512_mask_blend_ps(eq_mask,
        _mm512_mask_blend_ps(lt_mask, val_a, val_b),
        _mm512_setzero_ps());
    
    (out_idx_1, out_val_1, out_idx_2, out_val_2)
}
```

### 3.3 Test test Test

...

## 4. Integration and Optimization

### 4.1 Integration with Existing Framework

```rust
// src/accumulator/mod.rs

pub fn create_accumulator<T>(
    n_cols: usize, 
    dense_threshold: usize,
    config: &MagnusConfig,
) -> Box<dyn Accumulator<T>>
where
    T: Copy + Num + AddAssign + 'static,
{
    if n_cols <= dense_threshold {
        Box::new(dense::DenseAccumulator::new(n_cols))
    } else {
        // Check for AVX512 support
        #[cfg(target_arch = "x86_64")]
        {
            let features = avx512::Avx512Features::detect();
            if features.has_avx512f && config.enable_avx512 {
                return avx512::create_avx512_accumulator(
                    std::cmp::min(n_cols / 10, 1024),
                    &features,
                );
            }
        }
        
        // Fallback to generic sort accumulator
        let initial_capacity = std::cmp::min(n_cols / 10, 1024);
        Box::new(sort::SortAccumulator::new(initial_capacity))
    }
}
```

### 4.2 Configuration Extensions

```rust
// src/matrix/config.rs

pub struct MagnusConfig {
    // Existing fields...
    pub system_params: SystemParameters,
    pub dense_accum_threshold: usize,
    pub sort_method: SortMethod,
    pub enable_coarse_level: bool,
    pub architecture: Architecture,
    
    // New AVX512-specific fields
    pub enable_avx512: bool,
    pub avx512_sort_threshold: usize,  // Min size to use AVX512
    pub avx512_strategy: Avx512Strategy,
}

pub enum Avx512Strategy {
    SortThenReduce,      // Default strategy
    ModifiedCompareExchange,  // Experimental
    Adaptive,            // Choose based on input characteristics
}
```

### 4.3 Adaptive Strategy Selection

```rust
// src/accumulator/avx512/mod.rs

fn select_avx512_strategy(
    nnz_estimate: usize,
    duplicate_ratio: f32,
    config: &MagnusConfig,
) -> Avx512Strategy {
    match config.avx512_strategy {
        Avx512Strategy::Adaptive => {
            if duplicate_ratio > 0.3 {
                // High duplicate ratio favors modified compare-exchange
                Avx512Strategy::ModifiedCompareExchange
            } else if nnz_estimate < 256 {
                // Small inputs work well with bitonic networks
                Avx512Strategy::ModifiedCompareExchange
            } else {
                // Large, sparse inputs favor sort-then-reduce
                Avx512Strategy::SortThenReduce
            }
        }
        strategy => strategy,
    }
}
```

## Phase 5: Testing and Benchmarking (Weeks 10-12)

### 5.1 Unit Tests

```rust
// tests/avx512_accumulator.rs

#[cfg(target_arch = "x86_64")]
mod avx512_tests {
    use magnus::accumulator::avx512::*;
    
    #[test]
    fn test_avx512_sort_correctness() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("Skipping AVX512 test - not supported");
            return;
        }
        
        // Test data with known duplicates
        let col_indices = vec![5, 2, 8, 2, 5, 1, 8, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let mut acc = Avx512SortAccumulator::new(16);
        for (col, val) in col_indices.iter().zip(values.iter()) {
            acc.accumulate(*col, *val);
        }
        
        let (result_cols, result_vals) = acc.extract_result();
        
        // Expected: [1, 2, 3, 5, 8] with values [6.0, 6.0, 8.0, 6.0, 10.0]
        assert_eq!(result_cols, vec![1, 2, 3, 5, 8]);
        assert_eq!(result_vals, vec![6.0, 6.0, 8.0, 6.0, 10.0]);
    }
    
    #[test]
    fn test_compare_exchange_accumulation() {
        // Test modified compare-exchange with accumulation
        // ...
    }
}
```

### 5.2 Performance Benchmarks

```rust
// benches/avx512_performance.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use magnus::accumulator::*;

fn benchmark_accumulator_variants(c: &mut Criterion) {
    let sizes = vec![64, 256, 1024, 4096, 16384];
    let duplicate_ratios = vec![0.0, 0.1, 0.3, 0.5];
    
    let mut group = c.benchmark_group("avx512_accumulator");
    
    for size in &sizes {
        for ratio in &duplicate_ratios {
            let (indices, values) = generate_test_data(*size, *ratio);
            
            // Benchmark generic sort accumulator
            group.bench_with_input(
                BenchmarkId::new("generic", format!("size_{}_dup_{}", size, ratio)),
                &(&indices, &values),
                |b, (idx, val)| {
                    b.iter(|| {
                        let mut acc = sort::SortAccumulator::new(*size);
                        for (i, v) in idx.iter().zip(val.iter()) {
                            acc.accumulate(*i, *v);
                        }
                        black_box(acc.extract_result());
                    });
                },
            );
            
            // Benchmark AVX512 sort-then-reduce
            if is_x86_feature_detected!("avx512f") {
                group.bench_with_input(
                    BenchmarkId::new("avx512_sort", format!("size_{}_dup_{}", size, ratio)),
                    &(&indices, &values),
                    |b, (idx, val)| {
                        b.iter(|| {
                            let mut acc = avx512::Avx512SortAccumulator::new(*size);
                            for (i, v) in idx.iter().zip(val.iter()) {
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

criterion_group!(benches, benchmark_accumulator_variants);
criterion_main!(benches);
```

### 5.3 Integration Tests

```rust
// tests/magnus_with_avx512.rs

#[test]
fn test_magnus_spgemm_with_avx512() {
    let config = MagnusConfig {
        enable_avx512: true,
        avx512_strategy: Avx512Strategy::SortThenReduce,
        ..Default::default()
    };
    
    // Load test matrices
    let (a, b, expected_c) = load_test_matrices();
    
    // Compute with AVX512 enabled
    let c_avx512 = magnus_spgemm(&a, &b, &config);
    
    // Verify correctness
    assert_matrices_equal(&c_avx512, &expected_c, 1e-6);
    
    // Compare performance (optional)
    let config_generic = MagnusConfig {
        enable_avx512: false,
        ..Default::default()
    };
    
    let start = std::time::Instant::now();
    let _ = magnus_spgemm(&a, &b, &config_generic);
    let generic_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let _ = magnus_spgemm(&a, &b, &config);
    let avx512_time = start.elapsed();
    
    println!("Generic: {:?}, AVX512: {:?}, Speedup: {:.2}x",
        generic_time, avx512_time,
        generic_time.as_secs_f64() / avx512_time.as_secs_f64());
}
```

## Performance Targets and Metrics

### Expected Performance Improvements

Based on the MAGNUS paper and AVX512 capabilities:

1. **Sort-Then-Reduce**: 2-4x speedup over scalar implementation
   - AVX512 can process 16 32-bit integers per instruction
   - Efficient bitonic sorting networks for small arrays

2. **Modified Compare-Exchange**: 1.5-3x speedup for high-duplicate scenarios
   - Reduces memory bandwidth by combining sort and accumulation
   - Most beneficial when duplicate ratio > 30%

3. **Overall SpGEMM**: 1.3-2x speedup on suitable matrices
   - Improvement depends on proportion of sort-based rows
   - Greatest benefit for matrices with moderate sparsity

### Key Performance Indicators

1. **Throughput Metrics**
   - Elements sorted per second
   - Accumulations per second
   - Matrix multiplications per second

2. **Efficiency Metrics**
   - Instructions per element
   - Cache misses per accumulation
   - Memory bandwidth utilization

3. **Scalability Metrics**
   - Performance vs input size
   - Performance vs duplicate ratio
   - Performance vs sparsity pattern

## Risk Mitigation

### Technical Risks

1. **AVX512 Frequency Throttling**
   - Risk: AVX512 can cause CPU frequency reduction
   - Mitigation: Implement adaptive thresholds, only use for sufficiently large inputs

2. **Limited Hardware Support**
   - Risk: Not all Intel CPUs support AVX512
   - Mitigation: Runtime feature detection with fallback paths

3. **Complex Debugging**
   - Risk: SIMD code is harder to debug
   - Mitigation: Comprehensive unit tests, validation against scalar implementation

### Implementation Risks

1. **Integration Complexity**
   - Risk: Difficult to integrate with existing trait system
   - Mitigation: Careful API design, minimal changes to existing interfaces

2. **Performance Regression**
   - Risk: AVX512 might be slower for some inputs
   - Mitigation: Adaptive strategy selection, thorough benchmarking

## Success Criteria

1. **Correctness**: All tests pass with AVX512 enabled
2. **Performance**: Minimum 1.5x speedup on target workloads
3. **Compatibility**: Seamless fallback for non-AVX512 systems
4. **Maintainability**: Clean, documented code with minimal complexity increase

## Next Steps

1. Set up AVX512 development environment with appropriate hardware
2. Implement basic AVX512 sort kernel
3. Create microbenchmarks for algorithm validation
4. Begin integration with existing accumulator framework
5. Iterate based on performance measurements