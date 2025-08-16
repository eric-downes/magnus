# MAGNUS Benchmarks

## Quick Start

Run the quick tier benchmarks (< 30 seconds):
```bash
# Run quick tiered benchmarks
BENCH_TIER=quick cargo bench --bench tiered_benchmarks

# Run quick comparison against sprs
BENCH_TIER=quick cargo bench --bench comparison_benchmark
```

## Benchmark Organization

### Tiered Benchmarks (`tiered_benchmarks.rs`)
System benchmark with three tiers:
- **TIER 1 (Quick)**: < 30 seconds, small matrices, sanity checks
- **TIER 2 (Standard)**: < 5 minutes, medium matrices, regression testing  
- **TIER 3 (Stress)**: 10+ minutes, large matrices, stress testing

### Comparison Benchmark (`comparison_benchmark.rs`)
Direct comparison with `sprs` (standard Rust sparse matrix library):
- **Quick mode**: Fast comparison for CI/development (< 30 seconds)
- **Full mode**: Comprehensive comparison across sizes and patterns

### Specialized Benchmarks
- `accelerate_vs_neon.rs`: Apple Silicon accelerator comparison
- `arm_optimization.rs`: ARM-specific optimizations
- `matrix_multiply.rs`: Core SpGEMM operations
- `neon_diagnosis.rs`: NEON SIMD diagnostics
- `optimization_test.rs`: Accumulator optimizations
- `prefetch_benchmark.rs`: Memory prefetching effectiveness
- `quick_performance.rs`: Quick performance overview
- `suitesparse.rs`: SuiteSparse matrix collection tests

## Running Benchmarks

### Quick Comparison (Recommended for Development)
```bash
# Quick comparison vs. standard library
BENCH_TIER=quick cargo bench --bench comparison_benchmark

# Quick tiered benchmarks
BENCH_TIER=quick cargo bench --bench tiered_benchmarks
```

### Full Benchmarks
```bash
# Full comparison benchmark
cargo bench --bench comparison_benchmark

# Standard tier (medium workload)
BENCH_TIER=standard cargo bench --bench tiered_benchmarks

# Stress test (large workload)
BENCH_TIER=stress cargo bench --bench tiered_benchmarks
```

### Specific Benchmarks
```bash
# Run specific benchmark
cargo bench --bench matrix_multiply

# Run specific test within benchmark
cargo bench --bench comparison_benchmark -- small_matrices

# With custom parameters
cargo bench --bench comparison_benchmark -- --sample-size 10 --warm-up-time 1
```

## Understanding Results

### Performance Metrics
- **Time**: Median execution time per iteration
- **Throughput**: Elements processed per second
- **Outliers**: Statistical variance (see BENCHMARK_OUTLIERS_EXPLAINED.md)

### Expected Performance vs. sprs
| Matrix Size | Expected Speedup |
|------------|-----------------|
| Small (< 1K nnz) | ~1x (similar) |
| Medium (1K-100K nnz) | 1.5-3x faster |
| Large (> 100K nnz) | 2-5x faster |
| Many duplicates | 3-10x faster |

### Outliers
Criterion reports outliers as statistical anomalies:
- **Mild**: 1.5× IQR from quartiles
- **Severe**: 3× IQR from quartiles

For sparse matrix operations, 3-20% outliers are normal due to:
- Irregular memory access patterns
- Cache effects
- CPU frequency scaling
- Background processes

The meaningful metric is relative performance vs. alternatives.

## Benchmark Development

### Adding New Benchmarks
1. Create benchmark file in `benches/`
2. Add to `Cargo.toml`:
   ```toml
   [[bench]]
   name = "your_benchmark"
   harness = false
   ```
3. Use criterion for consistency

### Quick Tier Guidelines
Quick tier benchmarks should:
- Complete in < 30 seconds total
- Use small sample sizes (10-20)
- Test core functionality
- Include comparison with alternatives
- Provide rapid feedback for development

## Continuous Integration

For CI, use quick benchmarks:
```bash
# CI script
BENCH_TIER=quick cargo bench --bench comparison_benchmark -- --save-baseline main
BENCH_TIER=quick cargo bench --bench comparison_benchmark -- --baseline main
```

This provides fast feedback on performance regressions while keeping CI times reasonable.