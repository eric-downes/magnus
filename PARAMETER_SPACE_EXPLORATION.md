# Parameter Space Exploration Framework

## Overview

A comprehensive parameter space exploration framework has been built for MAGNUS that systematically generates test matrices across the full configuration space defined by the constants in `src/constants.rs`.

## Key Components

### 1. Parameter Groupings (`src/parameter_space.rs`)

Based on analysis of constants.rs, parameters are grouped into 5 interacting categories:

#### Group 1: Matrix Structure Parameters
- **Size**: 1K, 5K, 10K, 100K elements
- **Density**: 0.001 (ultra-sparse) to 0.2 (dense)
- **Average NNZ per row**: 1-1000

#### Group 2: Algorithm Selection Parameters
- **Dense threshold**: 128-512 elements
- **GPU threshold**: 1K-100K elements
- **SIMD minimum**: 4-32 elements

#### Group 3: Memory Hierarchy Parameters
- **L2 cache size**: 128KB-1MB
- **Chunk size**: 64-512 elements
- **Memory thresholds**: 2GB, 4GB, 8GB

#### Group 4: Prefetch Strategy Parameters
- **Distance threshold**: 1-4
- **Count multiplier**: 1-4
- **Hit rate thresholds**: 0.7, 0.9

#### Group 5: Accumulator Configuration
- **Initial capacity divisor**: 5-20
- **Max capacity**: 512-2048
- **Default size**: 128-512

### 2. Joint Distributions

The framework generates parameter combinations using:
- **Cartesian product** of discrete parameter values
- **Random sampling** within continuous ranges
- **Constraint checking** to eliminate invalid combinations

Total configurations generated: **6,912** covering the full parameter space

### 3. Matrix Generators

#### Random Generators (`MatrixGenerator`)
Generates NUM_RAND_OBJS (default: 3) matrices per configuration:
- **Ultra-sparse**: Controlled NNZ per row
- **Sparse**: Random distribution
- **Medium**: Balanced density
- **Dense**: High fill-in

#### Pattern Generators (`PatternMatrixGenerator`)
Specialized patterns for testing:
- **Banded**: Diagonal band structure (bandwidth 10-100)
- **Block Diagonal**: Block-structured matrices
- **Power Law**: Scale-free networks (α = 0.5-2.5)

#### SuiteSparse-Style Generators (`SuiteSparseStyleGenerator`)
Domain-specific matrices:
- **Circuit**: Small bandwidth, some dense rows
- **FEM**: Block structure from element connectivity
- **Web**: Power-law distribution, very sparse
- **Optimization**: Block angular with dense coupling rows

### 4. SuiteSparse Integration (`src/suitesparse_integration.rs`)

#### Matrix Market I/O
- **Read**: Parse .mtx files from SuiteSparse collection
- **Write**: Export generated matrices in Matrix Market format
- **Conversion**: Triplet to CSR format with duplicate handling

#### Recommended Test Matrices
Small to large matrices from different domains:
- `HB/bcsstk01`: 48×48 structural
- `Boeing/bcsstk13`: 2K×2K structural
- `SNAP/web-Stanford`: 282K×282K web graph
- `Williams/mac_econ_fwd500`: 207K×207K economic

### 5. Exploration Tool (`src/bin/explore_parameters.rs`)

Command-line tool with modes:
- **demo**: Show parameter space exploration
- **generate**: Create complete test suite
- **benchmark**: Performance testing across parameters
- **patterns**: Demonstrate pattern generation
- **suitesparse**: Generate domain-specific matrices

## Usage Examples

### Generate Test Suite
```bash
./target/debug/explore_parameters generate
```
Creates test matrices for all 6,912 parameter configurations.

### Run Benchmarks
```bash
./target/debug/explore_parameters benchmark
```
Benchmarks SpGEMM across parameter space (limited to smaller matrices).

### Generate Pattern Matrices
```bash
./target/debug/explore_parameters patterns
```
Creates banded, block diagonal, and power-law matrices.

### Generate SuiteSparse-Style Matrices
```bash
./target/debug/explore_parameters suitesparse
```
Creates circuit, FEM, web, and optimization matrices.

## Parameter Space Coverage

The framework ensures comprehensive testing by:

1. **Systematic Coverage**: All combinations of discrete parameters
2. **Random Sampling**: NUM_RAND_OBJS instances per configuration
3. **Constraint Validation**: Skip invalid parameter combinations
4. **Reproducible Generation**: Seeded RNG for consistent results

## Key Design Decisions

### Why These Groupings?

Parameters are grouped by their interaction patterns:
- Matrix structure affects algorithm selection
- Algorithm thresholds interact with memory hierarchy
- Memory parameters influence prefetch strategy
- All groups affect accumulator behavior

### Why NUM_RAND_OBJS = 3?

- Provides statistical variation without excessive computation
- Captures typical, best, and worst cases
- Manageable test suite size (≈20K total matrices)

### Integration with Constants

All parameter ranges are derived from `src/constants.rs`:
- Ensures consistency with implementation
- Automatic updates when constants change
- Type-safe parameter handling

## Test Matrix Statistics

Generated test suite characteristics:
- **Sizes**: 1K to 100K rows/columns
- **Densities**: 0.001 to 0.2
- **NNZ per row**: 1 to 1000
- **Total matrices**: ~20,000 (6,912 configs × 3 instances)
- **Patterns**: Banded, block, power-law, domain-specific

## Future Enhancements

1. **Adaptive Sampling**: Focus on interesting parameter regions
2. **Performance Prediction**: ML model for parameter impact
3. **Automatic Tuning**: Find optimal parameters for given hardware
4. **SuiteSparse Download**: Automatic fetching of real matrices
5. **Distributed Generation**: Parallel matrix generation for large suites

## Conclusion

This parameter space exploration framework provides:
- **Systematic testing** across all algorithm configurations
- **Reproducible generation** of test matrices
- **Domain-specific patterns** for realistic testing
- **Integration** with standard matrix formats
- **Extensible design** for future parameter additions

The framework enables comprehensive performance analysis and optimization of the MAGNUS algorithm across its entire configuration space.