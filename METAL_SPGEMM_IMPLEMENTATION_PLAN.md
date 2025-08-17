# Metal SpGEMM Implementation Plan

## Current Status

### ✅ Implemented
- **Metal bitonic sort** for sorting column indices and values (metal_impl.rs)
- **Basic Metal infrastructure**: Device creation, command queue, shader compilation
- **Metal compute kernels** (metal_kernels.metal):
  - `bitonic_sort_step` - Parallel GPU sorting
  - `parallel_reduce_sum` - Parallel reduction
  - Partial `spgemm_row` kernel (counts non-zeros only)

### ❌ Not Implemented  
- Full SpGEMM multiplication on GPU
- Symbolic phase (structure computation)
- Numeric phase (value computation)
- Memory management for variable-size outputs
- Integration with main SpGEMM pipeline

## Architecture Overview

```
Current Flow:
1. CPU: SpGEMM row multiplication
2. CPU: Generate intermediate products
3. GPU: Sort products (IMPLEMENTED)
4. GPU: Accumulate duplicates (PARTIALLY)
5. CPU: Assemble final matrix

Target Flow:
1. GPU: Symbolic phase (compute structure)
2. GPU: Allocate exact output memory
3. GPU: Numeric phase (compute values)
4. GPU: Direct output to CSR format
```

## Detailed Implementation Plan

### Phase 0: TDD

-- Write the primary tests by which we can measure our progress
-- ignore them until that work is complete.

### Phase 1: Complete Sort-Accumulate Pipeline (1-2 weeks)

**1.0 TDD**

-- Review the following and ensure test coverage for high level and critical features.
-- ignore them until that work is complete.

**1.1 Fix Accumulation Kernel**
```metal
kernel void accumulate_sorted_products(
    constant uint* sorted_indices [[buffer(0)]],
    constant float* sorted_values [[buffer(1)]],
    device uint* unique_indices [[buffer(2)]],
    device float* accumulated_values [[buffer(3)]],
    device atomic_uint* unique_count [[buffer(4)]],
    constant uint& n_elements [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    // Parallel scan to identify unique indices
    // Segmented reduction for accumulation
}
```

**1.2 Implement Parallel Scan**
- Use Blelloch scan algorithm for prefix sum
- Identify segment boundaries (where indices change)
- Compute output positions

**1.3 Memory Management**
- Implement two-pass approach:
  1. Predict using our zero-cost formula:
  `buffer_size = expected_products / duplicate_ratio * safety_factor`
  2. Choose GPU kernel based on prediction:
    - >75% duplicates: Hash-based accumulation in threadgroup memory
    - 50-75% duplicates: Segmented sort with compaction
    - <50% duplicates: Streaming output with atomic writes
  3. Single-pass processing with the appropriate accumulation method
  4. Allocate and fill output

### Phase 2: Symbolic SpGEMM (2-3 weeks)

**2.0 TDD**

-- Review the following and ensure test coverage for high level and critical features.
-- ignore them until that work is complete.

**2.1 Row-wise Non-zero Counter**
```metal
kernel void count_nnz_per_row(
    constant CSRMatrix& A,
    constant CSRMatrix& B,
    device uint* c_row_nnz,
    uint row_id [[thread_position_in_grid]])
{
    // Use hash table or sorting to count unique column indices
    // Output: number of non-zeros in each row of C
}
```

**2.2 Parallel Prefix Sum for Row Pointers**
```metal
kernel void compute_row_pointers(
    constant uint* row_nnz,
    device uint* row_ptr,
    constant uint& n_rows)
{
    // Exclusive scan to compute row_ptr array
}
```

**2.3 Memory Allocation Strategy**
- Pre-allocate pessimistic upper bound
- Use atomic counters for actual usage
- Compact in post-processing

### Phase 3: Numeric SpGEMM (3-4 weeks)

**3.0 TDD**

-- Review the following and ensure test coverage for high level and critical features.
-- ignore them until that work is complete.

**3.1 Full SpGEMM Kernel**
```metal
kernel void spgemm_numeric(
    constant CSRMatrix& A,
    constant CSRMatrix& B,
    device CSRMatrix& C,
    constant uint& row_id)
{
    // Per-row computation with local accumulation
    // Options:
    // 1. Hash table in shared memory (small rows)
    // 2. Sorting + merge (medium rows)  
    // 3. Dense vector (if enough shared memory)
}
```

**3.2 Load Balancing**
- **Dynamic scheduling**: Work-stealing queue for rows
- **Row binning**: Group rows by workload
  - Small rows: One thread per row
  - Medium rows: Warp/wavefront per row
  - Large rows: Block per row

**3.3 Memory Access Optimization**
- **Texture memory** for B matrix (random access pattern)
- **Shared memory** for frequently accessed A rows
- **Coalesced writes** to C matrix

### Phase 4: Advanced Optimizations (2-3 weeks)

**4.0 TDD**

-- Review the following and ensure test coverage for high level and critical features.
-- ignore them until that work is complete.

**4.1 Hybrid CPU-GPU Execution**
```cpp
class HybridSpGEMM {
    void execute(CSR& A, CSR& B, CSR& C) {
        // Partition rows by estimated work
        auto [cpu_rows, gpu_rows] = partition_work(A, B);
        
        // Async GPU execution
        auto gpu_future = async_gpu_spgemm(gpu_rows);
        
        // CPU handles irregular rows
        cpu_spgemm_parallel(cpu_rows);
        
        // Merge results
        merge_results(gpu_future.get(), cpu_results);
    }
};
```

**4.2 Multi-GPU Support**
- Row-wise partitioning across GPUs
- Overlap computation and communication
- Use Metal's multi-GPU features on Mac Studio

**4.3 Compression and Sparsity**
- **Bit-packing** for indices when possible
- **Value compression** for patterns (e.g., all 1.0s)
- **Empty row/column skipping**

### Phase 5: Integration and Testing (1-2 weeks)

**5.0 TDD**

-- Review the following and ensure test coverage for high level and critical features.
-- ignore them until that work is complete.

**5.1 Threshold Tuning**
```cpp
struct MetalThresholds {
    size_t min_nnz = 10000;        // Minimum for GPU
    size_t min_flops = 1000000;    // Minimum operations
    float min_density = 0.001;      // Minimum density
    float max_density = 0.1;        // Maximum (use dense)
};
```

**5.2 Fallback Mechanism**
- Automatic fallback to CPU for:
  - Small matrices
  - Nearly dense matrices
  - Memory allocation failures
  - Kernel timeout

**5.3 Performance Monitoring**
```cpp
struct MetalMetrics {
    double kernel_time;
    double transfer_time;
    size_t memory_used;
    size_t flops_performed;
    
    bool should_use_gpu() {
        return kernel_time < cpu_estimate * 0.7;
    }
};
```

## Technical Challenges and Solutions

### Challenge 1: Variable Output Size
**Problem**: SpGEMM output size unknown before computation  
**Solution**: Two-phase approach with symbolic+numeric, or pessimistic allocation with compaction

### Challenge 2: Load Imbalance
**Problem**: Rows have vastly different work  
**Solution**: Dynamic scheduling with work stealing, row binning by complexity

### Challenge 3: Memory Bandwidth
**Problem**: SpGEMM is memory-bound  
**Solution**: Maximize cache reuse, use texture memory, compress indices

### Challenge 4: Accumulation Conflicts
**Problem**: Multiple threads writing to same output position  
**Solution**: Sort-based approach, atomic operations, or thread-local accumulation

## Testing Strategy

### Unit Tests
1. **Kernel tests**: Each Metal kernel individually
2. **Accumulator tests**: Sort and accumulate operations
3. **Memory tests**: Allocation and deallocation
4. **Edge cases**: Empty matrices, single element, all duplicates

### Integration Tests
1. **Small matrices**: Verify correctness against reference
2. **Large matrices**: Performance benchmarks
3. **Special structures**: Diagonal, triangular, banded
4. **Stress tests**: Maximum size, high density

### Performance Benchmarks
```cpp
// Benchmark suite
void benchmark_metal_spgemm() {
    for (auto& matrix : test_matrices) {
        auto cpu_time = measure_cpu_spgemm(matrix);
        auto metal_time = measure_metal_spgemm(matrix);
        auto speedup = cpu_time / metal_time;
        
        log_performance(matrix.name, speedup);
    }
}
```

## Resource Requirements

### Development Time
- **Phase 1**: 1-2 weeks (accumulation completion)
- **Phase 2**: 2-3 weeks (symbolic phase)
- **Phase 3**: 3-4 weeks (numeric phase)
- **Phase 4**: 2-3 weeks (optimizations)
- **Phase 5**: 1-2 weeks (integration)
- **Total**: 9-14 weeks

### Hardware Requirements
- Mac with Apple Silicon (M1/M2/M3)
- Xcode with Metal developer tools
- Metal Performance Shaders framework
- Metal debugger and profiler

### Dependencies
- metal-rs crate (Rust Metal bindings)
- dispatch crate (Grand Central Dispatch)
- objc crate (Objective-C runtime)

## Success Metrics

### Performance Targets
- **Small matrices** (< 10K nnz): No regression vs CPU
- **Medium matrices** (10K - 1M nnz): 2-5x speedup
- **Large matrices** (> 1M nnz): 5-20x speedup
- **Memory efficiency**: < 2x peak memory vs CPU

### Quality Metrics
- **Correctness**: 100% match with reference implementation
- **Stability**: No crashes or hangs
- **Compatibility**: Works on all Apple Silicon Macs
- **Maintainability**: Clean, documented code

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Metal API changes | High | Version pinning, compatibility layer |
| Memory overflow | High | Careful bounds checking, fallback to CPU |
| Kernel timeout | Medium | Chunking large operations, watchdog |
| Precision issues | Medium | Double precision option, error bounds |
| Debugging difficulty | Low | Extensive logging, CPU reference mode |

## Next Steps

1. **Immediate** (This week):
   - Complete accumulation kernel
   - Add comprehensive Metal tests
   - Benchmark current implementation

2. **Short term** (Next month):
   - Implement symbolic phase
   - Design memory management
   - Create performance benchmarks

3. **Long term** (Quarter):
   - Complete numeric phase
   - Optimize for different matrix patterns
   - Integrate with production pipeline

## Conclusion

The Metal SpGEMM implementation is partially complete with working
sort/accumulate kernels. Full implementation requires significant
additional work but promises substantial performance gains for large
sparse matrices on Apple Silicon. The modular approach allows
incremental development and testing while maintaining system
stability.
