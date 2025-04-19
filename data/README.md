# Benchmark Data for MAGNUS

This directory contains data used for benchmarking the MAGNUS implementation.

## SuiteSparse Matrix Collection

The `suitesparse` subdirectory is intended for storing matrices from the SuiteSparse Matrix Collection. These matrices are used to benchmark the MAGNUS implementation against realistic sparse matrix problems.

### Downloading SuiteSparse Matrices

To use the SuiteSparse benchmarks, you need to download matrices in Matrix Market (.mtx) format from the SuiteSparse Matrix Collection:

1. Visit the SuiteSparse Matrix Collection website: https://sparse.tamu.edu/
2. Browse or search for matrices of interest
3. Download matrices in Matrix Market format
4. Place the .mtx files in the `suitesparse` subdirectory

### Recommended Matrices

The MAGNUS paper uses the following matrices for benchmarking, which we recommend using:

- cage4
- bcsstk08
- bcsstk09
- west0067
- dwt_234
- filter3D
- nemeth03
- OPF_6000
- bcsstk18
- msc10848
- Muu
- G3_circuit

### Running Benchmarks

Once you have downloaded the matrices, you can run the benchmarks with:

```bash
cargo bench --bench suitesparse
```

## Synthetic Matrices

The standard benchmarks also include synthetic matrices generated with controlled properties:

```bash
cargo bench --bench matrix_multiply
```

These don't require any external data files, as they are generated programmatically.