# MAGNUS Algorithm Notes

MAGNUS (Matrix Algebra on GPU and Multicore Systems) is an algorithm
for sparse general matrix-matrix multiplication (SpGEMM) that focuses
on improving memory access patterns through data reordering.  This
document contains detailed notes on the MAGNUS algorithm and our
implementation approach.

1. **Row Categorization**: Classifies matrix rows based on their computational requirements
2. **Fine-Level Reordering**: Improves data locality for intermediate-sized rows
3. **Coarse-Level Reordering**: Handles extremely large intermediate products
4. **Adaptive Accumulation**: Switches between dense and sort-based accumulation based on size

## 1. Row Categorization

Each row in matrix C is categorized based on its computational
characteristics, and routed to one of four accumulation algorithms

1. Dense Accumulator
2. Sort
3. Fine Level
4. Coarse Level

### Algorithm 1: Dense Accumulation

Standard approach using a dense accumulator array:
1. Initialize dense array of size n_cols
2. For each non-zero in row i of A, accumulate row j of B
3. Extract non-zeros from dense array

We use a dense array for value accumulation, a boolean flag array to
track non-zero positions, and column indices of non-zeros for
extraction.  There's also a `reset` capability for reusing allocated
memory.


### Algorithm 2: Sort-Based Accumulation

Alternative approach for small intermediate products:
1. Collect all column indices and values from intermediate products
2. Sort pairs by column index
3. Scan for duplicates and accumulate values

For sparse rows or when the dense array would be too large for cache,
we've implemented the sort-based accumulator.  Column indices and
values are stored as unsorted pairs.  We define abstract accumulator
raits for a common iface, and a size-based selector following the
paper's recommendation to use dense accumulators when they fit in
cache.  This should be memory-efficient for sparse intermediate
products.

### Algorithm 3: Fine-Level Reordering

Improves locality for moderate-sized intermediate products:
1. Compute histogram of column indices by chunk
2. Compute prefix sum to determine offsets
3. Reorder column indices and values by chunk
4. Process chunks sequentially for better cache utilization

We divide columns into chunks based on L2 cache size, use power-of-2
chunk sizes for efficient division via bit shifts, reordering
intermediate products by chunk for improved locality.  Chunks process
sequentially to maximize cache hits, with some special handling based
on hardware/arch.


### Algorithm 4: Coarse-Level Reordering

Handles extremely large intermediate products:
1. Extract subset of A for coarse rows
2. Convert to CSC format for column-oriented access
3. Process in batches for memory efficiency
4. Apply fine-level reordering within each batch

For rows with extremely large intermediate products, we've implemented
the coarse-level reordering `AHatCSC<T>`.  We process rows in batches,
preserving row identity while converting selected rows of A to CSC
format.  We use a merge step for combining multiple power-of-2 chunks
efficiently.  The paper leaves some implementation details that we
still need to address:

1. ~~**Coarse-Level Details**: Exact method for constructing AˆCSC~~ ✅
2. ~~**Coarse-Level Batch Size**: How to determine optimal batch size~~ ✅ 
3. **AVX-512 Sort**: Implementation of the AVX-512 sort with accumulation
4. **Parameter Tuning**: Finding optimal thresholds for different architectures


## Developments Notes

### Work So Far

1. **Phase 1**: Basic implementation with focus on correctness ✅
   - Implement core data structures (CSR/CSC) ✅
   - Simple row categorization ✅
   - Basic accumulator implementations (dense and sort-based) ✅

2. **Phase 2**: Complete MAGNUS algorithm (in progress)
   - Fine-level reordering algorithm ✅
   - Coarse-level reordering algorithm ✅
   - Integrated SpGEMM implementation ✅

3. **Phase 3**: Performance optimization
   - ARM / Apple
     - Hardware-specific optimizations ✅
     - Parameter tuning ✅
     - Advanced parallel execution
   - x86 / Linux
     - Hardware-specific optimizations
     - Parameter tuning
     - Advanced parallel execution
     
4. **Phase 4**: Product
   - Comprehensive API
   - Documentation
   - Performance benchmarking

### Performance Considerations

**Key performance factors**:

1. **Cache Utilization**: Optimizing for L1/L2/L3 cache usage
2. **SIMD Vectorization**: Leveraging AVX-512 and ARM NEON
3. **Parallelism**: Efficient thread utilization and work distribution
4. **Memory Access Patterns**: Minimizing random access and cache misses
5. **GPU Acceleration**: Potential for offloading specific operations to GPU

**Open Calibration questions**:

1. How do different row categorization thresholds affect performance?
2. What is the optimal chunk size for fine-level reordering?
3. How does performance scale with matrix density and size?
4. How significant is the performance impact of architecture-specific optimizations?
5. What are the tradeoffs between sort-based and dense accumulation?