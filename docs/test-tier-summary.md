# MAGNUS Test & Benchmark Tier System

## Overview

The MAGNUS project now has a comprehensive tiered testing and benchmarking system optimized for rapid development (TDD) while still providing thorough validation when needed.

## Test Tiers

### Quick Tests (TDD - <10 seconds)
```bash
./bench.sh test
# or directly:
cargo test --test large_matrix_quick --release
```

**Purpose**: Rapid feedback during development
**Duration**: 2-3 seconds
**Coverage**: 
- Matrices up to 1M non-zeros
- Basic correctness validation
- Serial vs parallel comparison
- Categorization testing

**When to use**: During active development, before every commit

### Large Matrix Tests (2+ minutes)
```bash
./bench.sh test-large
# or directly:
cargo test --test large_matrix --release -- --ignored
```

**Purpose**: Comprehensive validation with real-world sizes
**Duration**: 2-5 minutes
**Coverage**:
- 1M, 2M, 5M, 10M+ non-zero matrices
- Power-law distributions (graph workloads)
- Ultra-sparse matrices
- Memory scaling tests

**When to use**: Before merging to main, after optimization work

### All Tests (5+ minutes)
```bash
./bench.sh test-all
```

**Purpose**: Full validation including stress tests
**Duration**: 5+ minutes
**Coverage**: Everything including 20M+ nnz memory stress tests

## Benchmark Tiers

### Quick Benchmark (30 seconds)
```bash
./bench.sh quick
```
- Sanity check for performance regressions
- Small matrices (500x500 to 5Kx5K)
- Serial vs parallel comparison

### Standard Benchmark (2-3 minutes)
```bash
./bench.sh standard
```
- Normal use cases
- 0.5M to 1.5M non-zeros
- Comprehensive performance metrics

### Large Matrix Focus (5 minutes)
```bash
./bench.sh large
```
- **Primary use case testing**
- 2M to 3M+ non-zeros
- Various sparsity patterns
- Real-world matrix characteristics

### Full Suite (10+ minutes)
```bash
./bench.sh full
```
- Everything including stress tests
- Scaling studies
- 10M-20M non-zero tests

## Quick Reference

| Command | Duration | Purpose |
|---------|----------|---------|
| `./bench.sh test` | 3s | TDD - quick validation |
| `./bench.sh quick` | 30s | Catch performance regressions |
| `./bench.sh test-large` | 2-5min | Validate at scale |
| `./bench.sh large` | 5min | Benchmark primary use case |
| `./bench.sh full` | 10+min | Complete validation |

## Development Workflow

### During Development (TDD)
```bash
# Make changes
./bench.sh test           # 3 seconds - verify correctness
./bench.sh quick          # 30 seconds - check performance
```

### Before Commit
```bash
./bench.sh test           # Quick validation
./bench.sh standard       # 2-3 min comprehensive check
```

### Before Merge/Release
```bash
./bench.sh test-large     # Full correctness at scale
./bench.sh large          # Primary use case performance
```

### Major Changes
```bash
./bench.sh test-all       # All tests
./bench.sh full           # All benchmarks
```

## Performance Baselines

Save baseline before optimization:
```bash
BENCH_TIER=quick cargo bench --bench tiered_benchmark --save-baseline before
# Make changes
BENCH_TIER=quick cargo bench --bench tiered_benchmark --baseline before
```

For large matrix focus:
```bash
BENCH_TIER=large cargo bench --bench tiered_benchmark --save-baseline before
# Optimize for large matrices
BENCH_TIER=large cargo bench --bench tiered_benchmark --baseline before
```

## Matrix Characteristics Tested

| Size | Non-zeros | Density | Test Tier | Use Case |
|------|-----------|---------|-----------|----------|
| 2K×2K | ~100K | 2.5% | Quick | Algorithm validation |
| 5K×5K | ~250K | 1% | Quick | Parallel testing |
| 10K×10K | ~1M | 1% | Quick/Large | Medium scale |
| 15K×15K | ~2M | 0.9% | Large | Production size |
| 50K×50K | ~2-5M | 0.1% | Large | Large graphs |
| 100K×100K | ~2-10M | 0.02-0.1% | Large | Ultra-sparse |
| 200K×200K | ~20M | 0.05% | Stress | Memory limits |

## Key Improvements

1. **Fast TDD Loop**: Quick tests run in 3 seconds vs minutes
2. **Focused Testing**: Large matrix tests isolated from rapid development
3. **Clear Tiers**: Obvious when to use each tier
4. **Primary Use Case Focus**: Dedicated tier for >1M nnz matrices
5. **No Wasted Time**: Skip irrelevant tests when optimizing for large matrices