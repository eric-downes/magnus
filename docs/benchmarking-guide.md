# MAGNUS Benchmarking Guide

## Overview

The MAGNUS library includes a comprehensive benchmarking system designed to quickly validate performance improvements, especially for large sparse matrices (>1M non-zeros) which are the primary use case.

## Tiered Benchmark System

We use a tiered approach to balance quick feedback with comprehensive testing:

### Quick Tier (Default, ~30 seconds)
```bash
cargo bench --bench tiered_benchmark
# or explicitly:
BENCH_TIER=quick cargo bench --bench tiered_benchmark
```
- **Purpose**: Sanity check for major regressions
- **Duration**: ~30 seconds
- **Coverage**: Small matrices (500x500 to 5Kx5K)
- **Use when**: Making quick iterations, testing small changes

### Standard Tier (2-3 minutes)
```bash
BENCH_TIER=standard cargo bench --bench tiered_benchmark
```
- **Purpose**: Comprehensive normal use case testing
- **Duration**: 2-3 minutes
- **Coverage**: 0.5M to 1.5M non-zeros
- **Use when**: Before committing changes

### Large Tier (5 minutes)
```bash
BENCH_TIER=large cargo bench --bench tiered_benchmark
```
- **Purpose**: Focus on primary use case (large sparse matrices)
- **Duration**: ~5 minutes
- **Coverage**: 2M to 3M+ non-zeros, various sparsity patterns
- **Use when**: Optimizing for production workloads

### Full Tier (10+ minutes)
```bash
BENCH_TIER=full cargo bench --bench tiered_benchmark
```
- **Purpose**: Complete benchmark suite including stress tests
- **Duration**: 10+ minutes
- **Coverage**: All tests plus 10M-20M non-zero stress tests
- **Use when**: Major releases, architecture changes

## Running Large Matrix Tests

### Correctness Tests
```bash
# Run all large matrix correctness tests
cargo test --test large_matrix

# Run including memory-intensive tests
cargo test --test large_matrix -- --ignored

# Run a specific test
cargo test --test large_matrix test_large_matrix_correctness_1m
```

### Test Coverage
- `test_large_matrix_correctness_1m`: 1M non-zeros, basic validation
- `test_large_matrix_parallel_2m`: 2M non-zeros, parallel vs serial
- `test_power_law_matrix_5m`: 5M edges, graph-like structure  
- `test_ultra_sparse_10m`: 10M non-zeros, ultra-sparse (0.1% density)
- `test_memory_scaling_20m`: 20M+ non-zeros, memory stress test (ignored by default)

## Specialized Benchmarks

### Large Matrix Focus
```bash
cargo bench --bench large_matrix_bench
```
Includes:
- Scaling studies (how performance scales with size)
- Density studies (performance vs sparsity)
- Various matrix structures (uniform, power-law)

### Existing Benchmarks
```bash
# ARM optimizations
cargo bench --bench arm_optimization

# Accelerate vs NEON comparison
cargo bench --bench accelerate_vs_neon

# Prefetch effectiveness
cargo bench --bench prefetch_benchmark
```

## Interpreting Results

### Key Metrics
- **Throughput**: Non-zeros processed per second
- **Scaling**: Should be near-linear with thread count
- **Fill factor**: Output nnz / max(input nnz)

### Expected Performance

For large sparse matrices (>1M nnz):
- **Quick tier baseline**: Establishes baseline in seconds
- **Parallel speedup**: 3-8x on 8-core systems
- **Memory usage**: ~3-5x input matrix size during computation

### Performance Regression Detection

The quick tier is designed to catch regressions fast:
```bash
# Before changes
BENCH_TIER=quick cargo bench --bench tiered_benchmark --save-baseline before

# After changes  
BENCH_TIER=quick cargo bench --bench tiered_benchmark --baseline before
```

## Profiling Large Matrices

For detailed performance analysis:
```bash
# Profile with cargo-flamegraph
cargo flamegraph --bench tiered_benchmark -- --bench --profile-time 10

# Profile with samply (macOS)
samply record cargo bench --bench large_matrix_bench
```

## Best Practices

1. **Start with quick tier** for rapid iteration
2. **Run standard tier** before committing
3. **Use large tier** when optimizing for production
4. **Save baselines** before major changes
5. **Document unexpected results** in commit messages

## Matrix Characteristics

Our benchmarks cover various matrix types:

| Type | Density | NNZ/Row | Use Case |
|------|---------|---------|----------|
| Dense rows | 1-2% | 100-200 | Scientific computing |
| Medium | 0.1-0.5% | 50-100 | General sparse |
| Sparse | 0.05% | 20-50 | Large graphs |
| Ultra-sparse | <0.01% | 10-20 | Web graphs, social networks |

## Continuous Benchmarking

For CI/CD integration:
```yaml
# Quick check on every PR
test-quick:
  run: BENCH_TIER=quick cargo bench --bench tiered_benchmark

# Nightly comprehensive
test-nightly:
  run: BENCH_TIER=large cargo bench --bench tiered_benchmark
```