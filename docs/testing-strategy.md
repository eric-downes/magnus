# MAGNUS Testing Strategy

*Last updated: April 18, 2025*

This document outlines our approach to test-driven development for the MAGNUS implementation.

## Testing Philosophy

Our implementation follows strict test-driven development principles:
1. Write tests before implementing functionality
2. Focus on correctness before optimization
3. Maintain comprehensive test coverage throughout development
4. Use tests to validate against reference implementations and expected results

## Test Categories

### 1. Unit Tests

Unit tests verify the correctness of individual components:

- **Matrix Format Tests**: Validate CSR/CSC implementations and conversions
- **Algorithm Component Tests**: Verify each algorithmic building block
- **Utility Function Tests**: Test helper functions and data structures

### 2. Integration Tests

Integration tests verify that components work together correctly:

- **End-to-End SpGEMM Tests**: Test full matrix multiplication
- **Algorithm Integration Tests**: Verify fine/coarse level algorithm integration
- **Cross-Component Tests**: Test interaction between different modules
  
### 3. Property-Based Tests

Property-based tests verify mathematical properties and invariants:

- **Associativity Tests**: (A×B)×C = A×(B×C)
- **Distributivity Tests**: A×(B+C) = A×B + A×C
- **Identity Tests**: A×I = I×A = A

### 4. Performance Tests

Performance tests measure and verify computational efficiency:

- **Benchmark Suite**: Compare against baseline implementations
- **Scaling Tests**: Measure performance with increasing matrix sizes
- **Optimization Validation**: Verify that optimizations improve performance

## Test Matrices

We will use the following matrix types for testing:

1. **Simple Test Matrices**: Small, hand-crafted matrices with known results
2. **SuiteSparse Matrices**: Real-world sparse matrices from the SuiteSparse collection
3. **Synthetic Matrices**: Generated R-MAT and uniform random matrices
4. **Edge Case Matrices**: Matrices designed to test boundary conditions and special cases

## Testing Tools and Infrastructure

- **Framework**: Rust's built-in testing framework with cargo test
- **Benchmark Framework**: Criterion.rs for performance benchmarking
- **Reference Implementation**: We will implement a simple SpGEMM algorithm to serve as a reference
- **Matrix Generation**: Custom utilities for generating test matrices
- **Continuous Testing**: Run tests automatically during development

## Test-Driven Development Workflow

1. **Write Test**: Create a test for the functionality being implemented
2. **Verify Test Fails**: Ensure the test fails before implementation
3. **Implement Feature**: Write the minimum code to make the test pass
4. **Verify Test Passes**: Run the test to confirm implementation works
5. **Refactor**: Improve the implementation while ensuring tests continue to pass
6. **Repeat**: Move on to the next feature

## Test Documentation

Each test should include:
- Clear description of what is being tested
- Expected results and why they are correct
- Any specific edge cases being addressed
- Performance expectations, if applicable