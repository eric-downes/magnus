# MAGNUS Algorithm Notes

*Last updated: April 20, 2025*

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

## Implementation Details

### Dense Accumulator (Algorithm 1)

We've implemented the dense accumulator as described in the paper:

```rust
pub struct DenseAccumulator<T> {
    /// The dense accumulation array
    values: Vec<T>,
    
    /// Flags to track which positions in the dense array are non-zero
    occupied: Vec<bool>,
    
    /// Temporary storage for the column indices of non-zero elements
    col_indices: Vec<usize>,
}
```

Key features of our implementation:
- Uses a dense array for value accumulation
- Maintains a boolean flag array to track non-zero positions
- Stores column indices of non-zeros for efficient extraction
- Provides reset functionality to reuse allocated memory

### Sort-based Accumulator (Algorithm 2)

For sparse rows or when the dense array would be too large for cache, we've implemented the sort-based accumulator:

```rust
pub struct SortAccumulator<T> {
    /// Temporary storage for column indices of intermediate products
    col_indices: Vec<usize>,
    
    /// Temporary storage for values of intermediate products
    values: Vec<T>,
}
```

Key features of our implementation:
- Stores column indices and values as unsorted pairs
- Sorts indices and merges duplicate columns during extraction
- More memory-efficient for sparse intermediate products
- Provides functionality compatible with the Accumulator trait

### Accumulator Trait Design

We've designed a common trait interface for all accumulator types:

```rust
pub trait Accumulator<T>
where
    T: Copy + Num + AddAssign,
{
    /// Reset the accumulator to prepare for a new row
    fn reset(&mut self);
    
    /// Accumulate a single entry (column and value)
    fn accumulate(&mut self, col: usize, val: T);
    
    /// Extract the non-zero entries as sorted (column, value) pairs
    fn extract_result(self) -> (Vec<usize>, Vec<T>);
}
```

This design allows for:
- Polymorphic behavior with different accumulator implementations
- Factory function to select the appropriate accumulator type
- Future extensibility for other accumulator strategies

### Accumulator Selection Strategy

We've implemented a factory function that selects the appropriate accumulator:

```rust
pub fn create_accumulator<T>(n_cols: usize, dense_threshold: usize) -> Box<dyn Accumulator<T>>
where
    T: Copy + Num + AddAssign + 'static,
{
    if n_cols <= dense_threshold {
        Box::new(dense::DenseAccumulator::new(n_cols))
    } else {
        let initial_capacity = std::cmp::min(n_cols / 10, 1024);
        Box::new(sort::SortAccumulator::new(initial_capacity))
    }
}
```

The selection is based on:
- The number of columns in the output matrix
- A configurable threshold (default 256, which is a typical L2 cache line count)
- This aligns with the paper's recommendation to use dense accumulators when they fit in cache

## Reordering Strategies

### Fine-Level Reordering (Algorithm 3)

We've implemented the fine-level reordering strategy as described in the paper:

```rust
pub struct FineLevelReordering {
    /// Metadata about the chunk size and count
    metadata: ChunkMetadata,
}
```

Key features of our implementation:
- Divides columns into chunks based on L2 cache size
- Uses power-of-2 chunk sizes for efficient division via bit shifts
- Reorders intermediate products by chunk for improved locality
- Processes chunks sequentially to maximize cache hits
- Includes specialized handling for different matrix types and sizes

### Coarse-Level Reordering (Algorithm 4)

For rows with extremely large intermediate products, we've implemented the coarse-level reordering:

```rust
pub struct AHatCSC<T> {
    /// Maps to original rows in matrix A
    pub original_row_indices: Vec<usize>,
    
    /// The CSC representation of the matrix
    pub matrix: SparseMatrixCSC<T>,
}

pub struct CoarseLevelReordering {
    /// Metadata about the chunk size and count
    metadata: ChunkMetadata,
    
    /// Maximum batch size for processing
    batch_size: usize,
}
```

Key features of our implementation:
- Converts selected rows of A to CSC format for column-oriented access
- Processes rows in batches to maintain memory efficiency
- Reorders and processes each column chunk separately
- Maintains row identity mapping between original matrix and subset
- Implements the merge step for combining multiple chunks efficiently

### Chunking Metadata

Both reordering strategies rely on a common chunking approach:

```rust
pub struct ChunkMetadata {
    /// Size of each chunk (in elements)
    pub chunk_length: usize,
    
    /// Number of chunks
    pub n_chunks: usize,
    
    /// Number of bits to shift right to get chunk index (for power-of-2 chunk sizes)
    pub shift_bits: usize,
}
```

Key features:
- Uses power-of-2 chunk sizes for fast division via bit shifts
- Adapts to cache size for optimal memory usage
- Includes helper methods for efficient chunk manipulation

## Implementation Gaps

The paper leaves some implementation details that we still need to address:

1. ~~**Coarse-Level Details**: Exact method for constructing AˆCSC~~ ✅
2. ~~**Coarse-Level Batch Size**: How to determine optimal batch size~~ ✅ 
3. **AVX-512 Sort**: Implementation of the AVX-512 sort with accumulation
4. **Parameter Tuning**: Finding optimal thresholds for different architectures

## Our Implementation Approach

We are implementing the algorithm in phases:

1. **Phase 1**: Basic implementation with focus on correctness ✅
   - Implement core data structures (CSR/CSC) ✅
   - Simple row categorization ✅
   - Basic accumulator implementations (dense and sort-based) ✅

2. **Phase 2**: Complete MAGNUS algorithm (in progress)
   - Fine-level reordering algorithm ✅
   - Coarse-level reordering algorithm ✅
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