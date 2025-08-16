# Additions to the MAGNUS Algorithm

This document describes the high-level algorithmic and architectural
additions implemented beyond the base MAGNUS algorithm described in
the paper ["MAGNUS: Multi-level Accelerated GPU-Enabled Multigrid for
Numerically Unstructured Sparse Matrix
Multiplication"](https://arxiv.org/abs/2501.07056).

## 1. Adaptive Hardware-Accelerated Accumulation

### 1.1 Tiered Acceleration Strategy

The base MAGNUS algorithm uses a single accumulator approach. We use a
three-tier system that attempots an optimal accumulation
method based on array size and hardware capabilities:

- **Small arrays (≤32 elements)**: Hardware-specific SIMD networks
- **Medium arrays (33-9,999 elements)**: Platform-optimized libraries
- **Large arrays (≥10,000 elements)**: GPU acceleration

### 1.2 Platform-Specific SIMD Implementations

#### MacOS (Apple Silicon, ARM NEON, Metal GPU)

Integration with Apple's vDSP library for medium-sized arrays:
- Leverages `vDSP_vsorti` for optimized sorting with accumulation
- Provide 25% performance improvement over generic implementations (YMMV)

Implemented specialized bitonic sorting networks using ARM NEON intrinsics:
- 4-element networks using 6-comparison sorting
- 8, 16, and 32-element padded bitonic sorts
- Direct utilization of 128-bit NEON vector registers

Metal GPU acceleration for large accumulation operations:
- Bitonic sort kernels implemented in Metal Shading Language
- Thread-safe context management for multi-threaded SpGEMM
- Automatic fallback to CPU for smaller workloads

#### Linux x86

TBD

## 3. Architecture-Aware Memory Prefetching

### 3.1 Multi-Level Prefetch Strategies

Beyond basic memory access, the implementation provides sophisticated
prefetching with multiple strategies:

- **Conservative**: Prefetch only the next matrix row (< 64 bytes overhead)
- **Moderate**: Prefetch next row plus relevant B matrix rows (< 1KB overhead)
- **Aggressive**: Full lookahead prefetching (1-2KB overhead)
- **Adaptive**: Runtime adjustment based on observed cache hit patterns

### 3.2 Hardware-Specific Prefetch Instructions

The prefetching system uses architecture-specific instructions for
optimal cache utilization:

#### ARM64 Architecture
- L1,L2 cache hints: `PRFM PLDL1KEEP` for temporal, outer-loop data
- Streaming hints: `PRFM PLDL1STRM` for non-temporal access
- Write prefetch: `PRFM PSTL1KEEP` for output accumulation

#### x86-64 Architecture
- Multi-level hints using `_mm_prefetch` with T0, T1, and NTA flags
- Automatic translation of ARM-style hints to x86 equivalents

### 3.3 System Memory-Based Auto-Configuration

The prefetch strategy automatically adapts based on available system memory:
- Systems with <4GB RAM: Conservative prefetching only
- Systems with 4-8GB RAM: Moderate prefetching enabled
- Systems with >8GB RAM: Moderate prefetching with safety margins

## 4. Parallel Processing Enhancements

### 4.1 Coarse-Level Parallelization

Building on MAGNUS's row-level parallelism, the implementation adds:
- Batch processing of multiple rows per thread for better cache locality
- Dynamic load balancing based on row complexity prediction
- NUMA-aware thread scheduling on multi-socket systems

### 4.2 Fine-Grained Synchronization

For the fine-level reordering phase:
- Lock-free accumulation using atomic operations where beneficial
- Thread-local accumulation buffers to reduce contention
- Parallel histogram construction for chunk-based reordering

## 5. Algorithmic Refinements

### 5.1 Threshold Tuning

Apple Silicon's cache hierarchy favors smaller working sets due to its
unified memory arch and cache line size diff (128B vs. 64 on x86).
- Dense accumulator threshold: 192 for Apple Silicon (vs. 256 default)
- Chunk size: 512 for M-series cache architecture (vs. 1024 default)
- Coarse-level threshold: 10,000 for modern cache sizes

Hopefully this prevents L2 evictions, cache thrashing, etc.

## Summary

1. **Hardware Exploitation**: Leveraging platform-specific acceleration capabilities (NEON, Accelerate, Metal) while maintaining portability
2. **Memory Optimization**: Sophisticated prefetching and fused operations to maximize cache efficiency

The additions should maintain the benefits of the original MAGNUS
algorithm while providing significant practical performance
improvements, particularly on ARM-based systems.  The
hardware-specific optimizations appear to deliver 25-50% improvements
for Apple-Silocon.

