# MAGNUS Unified Test & Benchmark Tier System v2

## Overview

The MAGNUS project now features a unified tier system that integrates dimensional analysis (Buckingham π) testing with traditional matrix-based testing. This provides comprehensive coverage while maintaining reasonable execution times for different development phases.

## Key Innovation: Buckingham π Integration

Instead of testing all ~20,000 parameter combinations, we use dimensional analysis to test dimensionless groups (π-groups) that capture the fundamental behavior:

- **Full parameter space**: ~20,000 configurations
- **π-space (8 dimensions)**: 8,748 configurations  
- **Smart sampling**: 2,673 interesting configurations
- **Critical subset**: 5 essential configurations

## Tier Definitions

### Quick Tier (30 seconds)
**Purpose**: Rapid feedback during TDD  
**Use cases**: Pre-commit hooks, development iteration  
**π configurations**: 5 critical  
**Matrix sizes**: 500-5K  
**Command**: `./bench_v2.sh quick`

### Commit Tier (10 minutes)
**Purpose**: Validation before commits  
**Use cases**: Git hooks, CI on every push  
**π configurations**: 50 smart-sampled  
**Matrix sizes**: 1K-15K  
**Command**: `./bench_v2.sh commit`

### Pull Request Tier (30 minutes)
**Purpose**: Comprehensive PR validation  
**Use cases**: PR checks, branch merges  
**π configurations**: 500 interesting  
**Matrix sizes**: 5K-50K  
**Command**: `./bench_v2.sh pr`

### Release Tier (2 hours)
**Purpose**: Full validation for releases  
**Use cases**: Release candidates, major versions  
**π configurations**: 2,673 full exploration  
**Matrix sizes**: 10K-200K  
**Command**: `./bench_v2.sh release`

## Tier Comparison Table

| Tier | Time | π Configs | Matrix Sizes | Max NNZ | GPU | Memory Stress |
|------|------|-----------|--------------|---------|-----|---------------|
| Quick | 30s | 5 | 0.5-5K | 250K | No | No |
| Commit | 10min | 50 | 1-15K | 2M | Yes* | No |
| PR | 30min | 500 | 5-50K | 5M | Yes* | Yes |
| Release | 2hr | 2,673 | 10-200K | 20M | Yes* | Yes |

*GPU testing only on Apple Silicon

## The 5 Critical π Configurations

Each critical configuration represents a performance archetype:

1. **Cache-Optimal** (π₁=1.0): Perfect cache utilization
2. **SIMD-Optimal** (π₂=1.0): Maximum vectorization
3. **GPU-Transition** (π₅=10⁵): CPU/GPU boundary
4. **Memory-Bound** (π₃=10⁻⁵): Memory bandwidth saturation
5. **Sparse-Extreme** (π₈=0.1): Ultra-sparse handling

## Smart Sampling Strategy

The smart sampling (used in Commit/PR tiers) selects configurations meeting 3+ conditions:

- **Cache-critical**: |π₁ - 1.0| < 0.2
- **SIMD-boundary**: π₂ < 0.3 OR π₂ > 0.95  
- **Algorithm transition**: π₄ < 5e-4 AND π₅ > 5e4
- **Memory-constrained**: π₃ < 5e-5
- **Sparsity extreme**: π₈ < 0.15 OR π₈ > 8.0

This reduces 8,748 → ~2,673 configurations while maintaining coverage of interesting regions.

## Usage Examples

### Development Workflow

```bash
# During active development (TDD)
./bench_v2.sh quick          # 30 seconds

# Before committing
./bench_v2.sh commit         # 10 minutes

# Before creating PR
./bench_v2.sh pr            # 30 minutes

# Before release
./bench_v2.sh release       # 2 hours
```

### CI/CD Integration

```yaml
# GitHub Actions example
on: [push]
jobs:
  quick-test:
    runs-on: ubuntu-latest
    steps:
      - run: ./bench_v2.sh quick     # Every push

  commit-test:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: ./bench_v2.sh commit    # Main branch

  pr-test:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - run: ./bench_v2.sh pr        # Pull requests

  release-test:
    if: startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    steps:
      - run: ./bench_v2.sh release   # Tagged releases
```

### Environment Variables

```bash
# Set default tiers
export TEST_TIER=commit
export BENCH_TIER=commit

# Run with specific tier
BENCH_TIER=quick cargo bench --bench unified_benchmark
TEST_TIER=pr cargo test
```

## Rust API

```rust
use magnus::test_tiers::{TestTier, TierConfig, generate_test_matrices};

// Get current tier from environment
let tier = TestTier::from_env();

// Get configuration for tier
let config = TierConfig::for_tier(tier);

// Generate test matrices for a π configuration
let matrices = generate_test_matrices(tier, &pi_config);
```

## Benefits Over Previous System

### Previous System
- Fixed matrix sizes for each tier
- No systematic parameter exploration
- Redundant test coverage
- 5+ hours for comprehensive testing

### New Unified System
- Dimensionless π-groups ensure scale-invariant testing
- Smart sampling focuses on interesting regions
- 69% reduction in test count with better coverage
- Predictable time budgets for each tier
- Physical insights guide optimization

## Performance Expectations

Based on π-group analysis, expected scaling:

- **Base complexity**: O(n^1.5) for π₈=1.0
- **Cache effects** (π₁>1): +20% time
- **SIMD benefits** (π₂>0.5): -10% time
- **Memory pressure** (π₃<1e-4): +50% time

## Migration from Legacy Commands

| Legacy Command | New Command | Notes |
|----------------|-------------|-------|
| `./bench.sh test` | `./bench_v2.sh quick` | 30s instead of variable |
| `./bench.sh standard` | `./bench_v2.sh commit` | 10min cap |
| `./bench.sh large` | `./bench_v2.sh pr` | Better coverage |
| `./bench.sh full` | `./bench_v2.sh release` | Full π exploration |

## Tier Selection Guidelines

- **Use Quick**: When iterating on code, need immediate feedback
- **Use Commit**: Before any commit, ensures no regressions
- **Use PR**: Before merging, comprehensive validation
- **Use Release**: Before tagging/releasing, exhaustive testing

## Summary

The unified tier system provides:

1. **Predictable time budgets**: 30s to 2hr tiers
2. **Scientific coverage**: Buckingham π ensures testing fundamental behaviors
3. **Smart sampling**: Focus on interesting parameter regions
4. **Clear progression**: Quick → Commit → PR → Release
5. **Backward compatibility**: Legacy commands still work

This system replaces ad-hoc testing with a principled, dimensionally-aware approach that provides better coverage in less time.