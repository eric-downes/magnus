# MAGNUS Project Master Document

*Last updated: January 16, 2025*

## Project Overview

MAGNUS (Matrix Algebra on GPU and Multicore Systems) is an algorithm
for multiplying large sparse matrices, as described in [this
paper](https://arxiv.org/pdf/2501.07056). This project implements
MAGNUS in Rust with a focus on correctness, performance, and
cross-platform support.

## Current Status

**Current Phase**: Performance Optimization (Phase 3)

### ✅ Completed Components
- **Core Algorithm**: Full MAGNUS implementation with all four row categorization strategies
- **Matrix Formats**: CSR/CSC with bidirectional conversion utilities
- **Accumulator Methods**: Dense and sort-based with pluggable interface
- **Reordering Algorithms**: Fine-level and coarse-level locality generation
- **Parallel Execution**: Row-level and chunk-level parallelism via Rayon
- **Hardware Optimizations**: ARM NEON, Apple Accelerate, Metal GPU support
- **Prefetch System**: Smart prefetching for cache optimization
- **Test Suite**: 65+ unit tests, 20+ integration tests, 11 benchmarks

### 🚧 Active Development
- Performance benchmarking and parameter tuning
- Cross-platform optimization validation
- SuiteSparse matrix integration for real-world testing

### ⏳ Upcoming Work
- Intel AVX-512 optimizations
- Modified compare-exchange accumulator implementation
- Comprehensive performance documentation
- Production API stabilization

## Documentation Index

This document serves as the central reference for all project documentation. It will be updated regularly to reflect our current progress.

### Key Documents

1. [Project Roadmap](./docs/roadmap.md) - Overall implementation plan and timeline
2. [Development Journal](./docs/development-journal.md) - Chronological record of key decisions and milestones
3. [Testing Strategy](./docs/testing-strategy.md) - Approach to test-driven development
4. [Algorithm Notes](./docs/algorithm-notes.md) - Detailed notes on the MAGNUS algorithm and our implementation

### Documentation Guidelines

1. **Update Frequency**: 
   - This master document will be reviewed at the start of each development session
   - This master document will be updated at the end of each development session.
     - and at the start if discrepancies arise upon review.
   - Development journal entries will be added after significant progress or decisions

2. **Document Structure**:
   - Keep documents focused and specific
   - Use markdown headers and lists for easy navigation
   - Link between documents where appropriate
   - Include dates with all entries

3. **Progress Tracking**:
   - Current phase and status will be maintained in this document
   - Development journal will record detailed progress
   - Each document will include a "Last updated" date

## Development Environment

- **Language**: Rust (stable toolchain)
- **Platforms**: 
  - ✅ Hardware-agnostic baseline implementation
  - ✅ ARM/Apple Silicon (NEON, Accelerate, Metal)
  - 🚧 Intel x86-64 (AVX-512 planned)
  - ⏳ GPU acceleration (investigation phase)
- **Testing**: Comprehensive test suite with reference implementation validation
- **Key Dependencies**:
  - `rayon`: Parallel execution
  - `num-traits`: Generic numeric operations
  - `sprs`: Sparse matrix utilities for testing
  - `criterion`: Performance benchmarking
  - Platform-specific: Accelerate framework, Metal shaders

## Contact

This project is being developed as an open-source implementation under the MIT license.