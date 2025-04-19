# MAGNUS Dependencies

*Last updated: April 18, 2025*

This document lists and explains the external dependencies used in the MAGNUS implementation.

## Core Dependencies

### Matrix Libraries

| Dependency | Version | Description | Usage |
|------------|---------|-------------|-------|
| `sprs` | 0.11 | Sparse matrix representations and operations | Reference implementation and utility functions |
| `ndarray` | 0.15 | N-dimensional array operations | Dense matrix operations and accumulator |

### Parallelism

| Dependency | Version | Description | Usage |
|------------|---------|-------------|-------|
| `rayon` | 1.8 | Data-parallel computations | Parallel row processing and accumulation |

### Numeric Support

| Dependency | Version | Description | Usage |
|------------|---------|-------------|-------|
| `num-traits` | 0.2 | Numeric traits for generic programming | Generic matrix implementations |
| `aligned-vec` | 0.5 | Aligned memory allocation | SIMD-friendly memory allocation |

## Development Dependencies

### Testing and Benchmarking

| Dependency | Version | Description | Usage |
|------------|---------|-------------|-------|
| `criterion` | 0.5 | Benchmarking framework | Performance testing and optimization |
| `proptest` | 1.3 | Property-based testing | Validation of mathematical properties |

## Planned / Future Dependencies

### Hardware Optimizations

| Dependency | Status | Description | Usage |
|------------|--------|-------------|-------|
| `std::arch` | Core Rust | SIMD intrinsics | Low-level vectorization |
| CUDA/ROCm bindings | Planned | GPU acceleration | Offloading performance-critical operations |

### Advanced Features

| Dependency | Status | Description | Usage |
|------------|--------|-------------|-------|
| `openmp-sys` | Potential | OpenMP bindings | Alternative parallelization model |
| HDF5 bindings | Potential | Matrix I/O | Reading/writing large matrices |