# MAGNUS

MAGNUS (Matrix Algebra for Gigantic NUmerical Systems) is an algorithm
for multiplying large sparse matrices, as described in [this
paper](https://arxiv.org/pdf/2501.07056).

This implementation focuses on building a high-performance Rust version with:
1. Hardware-agnostic foundation first
2. Intel AVX-512 and ARM NEON optimizations
3. Potential GPU/CUDA acceleration for specific operations

## Project Status

This project is under active development following the [roadmap](./docs/roadmap.md).

## Features (Planned)

- Efficient sparse matrix multiplication (SpGEMM)
- Row-wise adaptive algorithms based on computational requirements
- Fine and coarse-level reordering for improved memory locality
- Hardware-specific optimizations for Intel and ARM architectures
- GPU acceleration for performance-critical operations

## Documentation

- [Project Roadmap](docs/roadmap.md)
- [Development Documentation](docs/master-document.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
