# MAGNUS Project Master Document

*Last updated: April 19, 2025*

## Project Overview

MAGNUS (Matrix Algebra on GPU and Multicore Systems) is an algorithm
for multiplying large sparse matrices, as described in [this
paper](https://arxiv.org/pdf/2501.07056). This project aims to
implement MAGNUS in Rust, focusing first on correctness and then on
performance.

## Current Status

**Current Phase**: Core Implementation (Phase 1)
- Established project roadmap and documentation
- Set up Rust project structure with dependencies
- Implemented basic matrix formats (CSR/CSC) with format conversion utilities
- Created reference SpGEMM implementation for testing
- Implemented row categorization logic
- ✅ Implemented dense accumulator for SpGEMM
- ✅ Implemented sort-based accumulator for SpGEMM
- Set up rustdoc documentation with GitHub Actions

**Next Steps**:
- Implement reordering strategies (fine-level and coarse-level)
- Create the main MAGNUS SpGEMM function with accumulator selection
- Implement row-parallel execution
- Conduct benchmarking against reference implementation

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

- **Language**: Rust
- **Platform**: Initially hardware-agnostic, with later optimizations for:
  - Intel (AVX-512 / SIMD)
  - ARM (Apple Silicon)
  - Potential GPU acceleration
- **Testing**: Comprehensive test suite with comparison to reference implementations
- **Dependencies**: See [dependencies list](/docs/dependencies.md) when created

## Contact

This project is being developed as an open-source implementation under the MIT license.