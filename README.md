# MAGNUS++

![Rust CI](https://github.com/eric/magnus/actions/workflows/rust-ci.yml/badge.svg)

MAGNUS++ (Matrix Algebra for Gigantic NUmerical Systems Plus-Plus) is
a high-performance Rust implementation of the sparse matrix
multiplication algorithm described in [Pou, Laukemann, & Patrini
(2025)](https://arxiv.org/pdf/2501.07056) with additions (mostly GPU
stuff) as described [here](./docs/additions_to_magnus.md).

## Features

- **Efficient Sparse Matrix Multiplication (SpGEMM)**: Optimized for large-scale sparse matrices
- **Adaptive Algorithm Selection**: Automatically chooses the best strategy based on matrix characteristics
- **Architecture-Specific Optimizations**: Detects and utilizes CPU features (AVX-512, ARM NEON)
- **Apple Accelerate Framework**: Automatically uses Apple's optimized libraries on macOS
- **Parallel Execution**: Multi-threaded processing using Rayon
- **Memory-Efficient**: Fine and coarse-level reordering for improved cache locality
- **Large Sparse Support**: As of Aug 2025, nnz ~ 57M working.

## Installation

Add MAGNUS to your `Cargo.toml`:

```toml
[dependencies]
magnus = { git = "https://github.com/eric/magnus" }
```

## Usage

### Basic Example

```rust
use magnus::{SparseMatrixCSR, MagnusConfig, magnus_spgemm};

// Create sparse matrices in CSR format
let a = SparseMatrixCSR::<f64>::new(
    3, 3,                        // 3x3 matrix
    vec![0, 2, 4, 6],           // row pointers
    vec![0, 1, 1, 2, 0, 2],     // column indices
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // values
);

let b = SparseMatrixCSR::<f64>::new(
    3, 3,
    vec![0, 2, 4, 6],
    vec![1, 2, 0, 2, 0, 1],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
);

// Multiply matrices using MAGNUS
let config = MagnusConfig::default();
let c = magnus_spgemm(&a, &b, &config);
```

### Parallel Execution

For large matrices, use the parallel version:

```rust
use magnus::{magnus_spgemm_parallel, MagnusConfig};

let config = MagnusConfig::default();
let c = magnus_spgemm_parallel(&a, &b, &config);
```

### Architecture-Specific Configuration

MAGNUS automatically detects your CPU architecture, but you can specify it manually:

```rust
use magnus::{Architecture, MagnusConfig};

// For Apple Silicon (ARM NEON)
let config = MagnusConfig::for_architecture(Architecture::ArmNeon);

// For Intel with AVX-512
let config = MagnusConfig::for_architecture(Architecture::X86WithAVX512);
```

## Testing

### Run All Tests

```bash
cargo test
```

### Run Specific Test Categories

```bash
# Unit tests only
cargo test --lib

# Integration tests
cargo test --test arm_integration
cargo test --test spgemm_correctness

# Architecture-specific tests (ARM/Apple Silicon)
cargo test --test arm_neon

# Release mode (optimized)
cargo test --release
```

## Benchmarking

MAGNUS includes a comprehensive benchmarking suite with multiple tiers
for different use cases.

TLDR: fast script!
```bash
./bench.sh test      # ~3s Run minimal correctness tests
./bench.sh           # ~30s standard sanity checks
./bench.sh large     # ~5 min workout, large matrix focus
./bench.sh standard  # good practice before commit
```

### Large Matrix Benching

Before making changes
```bash
BENCH_TIER=quick cargo bench --bench tiered_benchmark --save-baseline before
```

After changes - compare
```
BENCH_TIER=quick cargo bench --bench tiered_benchmark --baseline before
```

When optimizing for production -- hit yout PR!
```
BENCH_TIER=large cargo bench --bench tiered_benchmark
```

### Tiered Benchmark System

The benchmark system has three tiers to balance between quick feedback
and comprehensive testing:

#### Tier 1: Quick Benchmarks (< 30 seconds)
For rapid development feedback and CI/CD:

```bash
BENCH_TIER=quick cargo bench --bench tiered_benchmarks
```

- Matrices: 50x50 to 500x500
- Includes pre-flight sanity checks
- Suitable for development iteration

#### Tier 2: Standard Benchmarks (< 5 minutes)
For performance validation and regression testing:

```bash
BENCH_TIER=standard cargo bench --bench tiered_benchmarks
```

- Matrices: 1,000x1,000 to 5,000x5,000
- Tests both serial and parallel execution
- Good for pre-commit validation

#### Tier 3: Stress Tests (10+ minutes)
For finding performance limits with real-world workloads:

```bash
BENCH_TIER=stress cargo bench --bench tiered_benchmarks
```

- Matrices: 10,000x10,000 to 100,000x100,000
- High memory usage (check available RAM)
- Comprehensive performance analysis

### Specialized Benchmarks

#### ARM/Apple Silicon Optimization
```bash
# Test ARM NEON optimizations
cargo bench --bench arm_optimization

# Quick performance comparison
cargo bench --bench quick_performance
```

#### Matrix Operations
```bash
# Test individual matrix operations
cargo bench --bench matrix_multiply
```

#### NEON Diagnostics (Apple Silicon)
```bash
# Detailed NEON performance analysis
cargo bench --bench neon_diagnosis
```

### Running Specific Benchmarks

You can run specific benchmark groups:

```bash
# Run only accumulator benchmarks
cargo bench --bench arm_optimization accumulator_methods

# Run only threshold tuning
cargo bench --bench quick_performance threshold
```

### Benchmark Output

Benchmarks are saved in `target/criterion/` with HTML reports showing:
- Performance over time
- Statistical analysis
- Comparison plots

## Performance

### Architecture Detection

MAGNUS automatically detects and optimizes for your CPU:

| Architecture | Detection | Optimizations |
|-------------|-----------|---------------|
| Apple Silicon (M1/M2/M3) | ✅ Automatic | Accelerate framework (default), ARM NEON, tuned thresholds |
| Intel x86 with AVX-512 | ✅ Automatic | AVX-512 sorting |
| Intel x86 without AVX-512 | ✅ Automatic | AVX2 optimizations |
| Generic | ✅ Fallback | Portable implementation |

#### Apple Silicon Optimization

On Apple Silicon, MAGNUS defaults to using Apple's Accelerate
framework for optimal performance. To use pure NEON implementation
instead:

```bash
# Disable Accelerate and use NEON-only implementation
MAGNUS_DISABLE_ACCELERATE=1 cargo run --release
```

### Performance Tips

1. **Matrix Size**: Performance scales well with parallel execution for matrices > 1000x1000
2. **Density**: Optimized for sparse matrices (< 10% density)
3. **Memory**: Ensure sufficient RAM for large matrices (estimate: rows × nnz_per_row × 16 bytes)
4. **Threads**: Defaults to all CPU cores; adjust with `RAYON_NUM_THREADS` environment variable

## Algorithm Details

MAGNUS uses one of four strategies based on row characteristics:

1. **Sort-based**: For small intermediate products
2. **Dense Accumulation**: When intermediate products fit in L2 cache
3. **Fine-level Reordering**: For improved memory locality
4. **Coarse-level Reordering**: For very large intermediate products

The algorithm automatically selects the optimal strategy for each matrix row.

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/eric-downes/magnuspp
cd magnus

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run benchmarks
BENCH_TIER=quick cargo bench --bench tiered_benchmarks
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Documentation

- [Algorithm Details](docs/master-document.md)
- [Testing Strategy](docs/testing-strategy.md)
- [ARM Optimization](docs/arm-hardware.md)
- [Performance Report](PERFORMANCE_REPORT.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
