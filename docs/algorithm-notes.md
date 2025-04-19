# MAGNUS Algorithm Notes

*Last updated: April 18, 2025*

This document contains detailed notes on the MAGNUS algorithm and our implementation approach. It serves as a reference for key algorithmic components and implementation decisions.

## Algorithm Overview

MAGNUS (Matrix Algebra on GPU and Multicore Systems) is an algorithm for sparse general matrix-matrix multiplication (SpGEMM) that focuses on improving memory access patterns through data reordering.

### Key Innovations

1. **Row Categorization**: Classifies matrix rows based on their computational requirements
2. **Fine-Level Reordering**: Improves data locality for intermediate-sized rows
3. **Coarse-Level Reordering**: Handles extremely large intermediate products
4. **Adaptive Accumulation**: Switches between dense and sort-based accumulation based on size

## Row Categorization

The algorithm categorizes each row in matrix C based on its computational characteristics:

1. **Sort**: Small intermediate products that can be efficiently processed with sorting
2. **Dense Accumulation**: Intermediate products that fit in L2 cache
3. **Fine-Level**: Requires fine-level chunking for locality
4. **Coarse-Level**: Requires both coarse and fine-level reordering

## Key Algorithms

### Algorithm 1: Dense Accumulation

Standard approach using a dense accumulator array:
1. Initialize dense array of size n_cols
2. For each non-zero in row i of A, accumulate row j of B
3. Extract non-zeros from dense array

### Algorithm 2: Sort-Based Accumulation

Alternative approach for small intermediate products:
1. Collect all column indices and values from intermediate products
2. Sort pairs by column index
3. Scan for duplicates and accumulate values

### Algorithm 3: Fine-Level Reordering

Improves locality for moderate-sized intermediate products:
1. Compute histogram of column indices by chunk
2. Compute prefix sum to determine offsets
3. Reorder column indices and values by chunk
4. Process chunks sequentially for better cache utilization

### Algorithm 4: Coarse-Level Reordering

Handles extremely large intermediate products:
1. Extract subset of A for coarse rows
2. Convert to CSC format for column-oriented access
3. Process in batches for memory efficiency
4. Apply fine-level reordering within each batch

## Implementation Gaps

The paper leaves several implementation details unspecified:

1. **Coarse-Level Details**: Exact method for constructing AˆCSC
2. **Accumulator Switch Threshold**: When to switch between sort and dense accumulators
3. **Coarse-Level Batch Size**: How to determine optimal batch size
4. **AVX-512 Sort**: Implementation of the AVX-512 sort with accumulation

## Our Implementation Approach

We will implement the algorithm in phases:

1. **Phase 1**: Basic implementation with focus on correctness
   - Implement core data structures (CSR/CSC)
   - Simple row categorization
   - Basic accumulator implementations

2. **Phase 2**: Complete MAGNUS algorithm
   - Fine-level reordering algorithm
   - Coarse-level reordering algorithm
   - Integrated SpGEMM implementation

3. **Phase 3**: Performance optimization
   - Hardware-specific optimizations
   - Parameter tuning
   - Advanced parallel execution

4. **Phase 4**: Productionization
   - Comprehensive API
   - Documentation
   - Performance benchmarking

## Performance Considerations

Key performance factors to consider:

1. **Cache Utilization**: Optimizing for L1/L2/L3 cache usage
2. **SIMD Vectorization**: Leveraging AVX-512 and ARM NEON
3. **Parallelism**: Efficient thread utilization and work distribution
4. **Memory Access Patterns**: Minimizing random access and cache misses
5. **GPU Acceleration**: Potential for offloading specific operations to GPU

## Research Questions

Open questions to investigate during implementation:

1. How do different row categorization thresholds affect performance?
2. What is the optimal chunk size for fine-level reordering?
3. How does performance scale with matrix density and size?
4. How significant is the performance impact of architecture-specific optimizations?
5. What are the tradeoffs between sort-based and dense accumulation?