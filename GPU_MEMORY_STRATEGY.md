# GPU Memory Strategy for Metal SpGEMM

## The Problem with Two-Pass Counting

The original plan suggested:
1. Count unique elements (requires full scan)
2. Allocate exact memory
3. Fill output

This is **inefficient** because:
- Extra memory pass (slow)
- Synchronization barrier between passes
- Defeats purpose of GPU parallelism

## Better Approach: Leverage Our Zero-Cost Prediction

### What We Already Have (from FUSED_OPERATIONS_SUMMARY)

We built a **zero-cost duplicate prediction** that uses information already available during SpGEMM:

```rust
// During row multiplication, we already compute:
expected_products = sum(b_row_sizes for each a_col)
duplicate_ratio = expected_products / b_n_cols

// If ratio > 4, expect >75% duplicates
```

### GPU Strategy Using This Insight

#### Option 1: Pessimistic Allocation with Streaming Compaction

```metal
// 1. Predict output size using our zero-cost formula
predicted_nnz = expected_products / predicted_duplicate_ratio;
buffer_size = predicted_nnz * SAFETY_FACTOR; // e.g., 1.5x

// 2. Allocate once
device buffer<uint> output_indices[buffer_size];
device buffer<float> output_values[buffer_size];
device atomic_uint write_pos = 0;

// 3. Stream results with atomic writes
kernel void accumulate_products(...) {
    // Each thread handles a segment
    local_results = sort_and_accumulate_local(segment);
    
    // Atomic reserve space in output
    uint my_pos = atomic_fetch_add(&write_pos, local_results.count);
    
    // Direct write to final position
    output_indices[my_pos...] = local_results.indices;
    output_values[my_pos...] = local_results.values;
}
```

**Pros:**
- Single allocation
- No synchronization barrier
- Streaming output

**Cons:**
- May over-allocate (but prediction is good)
- Atomic contention for write position

#### Option 2: Segmented Processing with Local Buffers

```metal
// Based on duplicate prediction, choose segment size
if (high_duplicates_predicted) {
    segment_size = 256;  // Smaller segments, more accumulation
    local_buffer_size = 64;  // Expect 75% reduction
} else {
    segment_size = 1024;  // Larger segments
    local_buffer_size = 512;  // Less reduction expected
}

kernel void spgemm_row_segmented(
    threadgroup float local_values[LOCAL_BUFFER_SIZE],
    threadgroup uint local_indices[LOCAL_BUFFER_SIZE],
    device atomic_uint segment_offsets[NUM_SEGMENTS]
) {
    // Each threadgroup processes a segment
    uint segment_id = threadgroup_id;
    
    // Process segment into threadgroup memory
    local_nnz = process_segment_locally(segment_id);
    
    // Prefix sum to get output positions
    if (thread_id == 0) {
        output_offset = atomic_fetch_add(&segment_offsets[segment_id], local_nnz);
    }
    
    // Parallel write to final position
    parallel_copy_to_device(output_offset, local_results);
}
```

**Pros:**
- Leverages threadgroup memory (fast)
- Minimal atomic operations
- Adaptive to prediction

**Cons:**
- Fixed threadgroup size limits
- May need multiple kernel launches

#### Option 3: Hash-Based Accumulation (Best for High Duplicates)

When we predict >75% duplicates, use GPU hash table:

```metal
// Per-row hash table in shared memory
threadgroup struct Entry {
    uint col_idx;
    float value;
    uint next;  // Chain for collisions
} hash_table[HASH_SIZE];

kernel void spgemm_row_hashed(
    constant uint* a_cols,
    constant float* a_vals,
    constant CSR* B,
    device CSR* C
) {
    // Initialize threadgroup hash table
    if (thread_id < HASH_SIZE) {
        hash_table[thread_id] = EMPTY_ENTRY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each thread processes some products
    for (uint i = thread_id; i < num_products; i += threads_per_group) {
        uint col = compute_product_column(i);
        float val = compute_product_value(i);
        
        // Hash insert with accumulation
        uint hash = col % HASH_SIZE;
        atomic_add_to_hash(&hash_table[hash], col, val);
    }
    
    // Extract unique entries from hash table
    if (thread_id == 0) {
        extract_hash_to_output(hash_table, output);
    }
}
```

**Pros:**
- Direct accumulation (no sort needed)
- Excellent for high duplicate scenarios
- Matches our CPU fused approach

**Cons:**
- Limited by threadgroup memory size
- Hash collisions need handling

### Recommended Approach

Based on our zero-cost prediction:

```cpp
struct GPUStrategy {
    static Strategy select(const DuplicateContext& ctx) {
        float duplicate_ratio = ctx.expected_ratio();
        
        if (duplicate_ratio > 4.0) {
            // >75% duplicates expected
            return Strategy::HashAccumulation;
        } else if (duplicate_ratio > 2.0) {
            // 50-75% duplicates
            return Strategy::SegmentedWithCompaction;
        } else {
            // Few duplicates
            return Strategy::StreamingOutput;
        }
    }
    
    static size_t predict_buffer_size(const DuplicateContext& ctx) {
        // Use our zero-cost prediction
        size_t expected_unique = ctx.expected_products / max(1, ctx.duplicate_ratio);
        
        // Add safety margin based on confidence
        float safety = (ctx.duplicate_ratio > 4.0) ? 1.2 : 1.5;
        return expected_unique * safety;
    }
};
```

### Integration with Existing Code

Our existing duplicate prediction integrates perfectly:

```rust
// CPU side - already implemented
let context = DuplicateContext::gather_during_spgemm(a_row, b);

// Pass to GPU
let gpu_strategy = match context.duplicate_ratio() {
    r if r > 4.0 => GpuStrategy::Hashed,
    r if r > 2.0 => GpuStrategy::Segmented,
    _ => GpuStrategy::Streaming,
};

// Allocate based on prediction
let buffer_size = context.predict_unique_count() * SAFETY_FACTOR;
let gpu_buffer = metal_device.new_buffer(buffer_size);

// Launch appropriate kernel
match gpu_strategy {
    GpuStrategy::Hashed => launch_hashed_kernel(...),
    GpuStrategy::Segmented => launch_segmented_kernel(...),
    GpuStrategy::Streaming => launch_streaming_kernel(...),
}
```

## Memory Allocation Summary

Instead of costly two-pass counting:

1. **Use zero-cost prediction** from our existing system
2. **Allocate once** with intelligent safety margin
3. **Choose kernel strategy** based on predicted duplicates:
   - High duplicates (>75%): Hash-based accumulation
   - Medium duplicates (50-75%): Segmented with compaction
   - Low duplicates (<50%): Streaming output
4. **Single-pass processing** with strategy-appropriate accumulation

This approach:
- ✅ No extra memory passes
- ✅ Leverages our existing duplicate prediction
- ✅ Adaptive to workload characteristics
- ✅ Minimal synchronization overhead
- ✅ Matches CPU optimization strategy

## Benchmarking Targets

For GPU implementation validation:

| Scenario | Duplicates | Strategy | Expected Speedup vs CPU |
|----------|------------|----------|------------------------|
| Dense×Dense | >75% | Hashed | 5-10× |
| Typical Sparse | 20-50% | Segmented | 3-5× |
| Very Sparse | <20% | Streaming | 2-3× |
| Small Matrices | Any | CPU | No GPU (overhead too high) |

## Conclusion

By leveraging our zero-cost duplicate prediction system, we can:
1. Avoid expensive two-pass counting
2. Choose optimal GPU strategy upfront
3. Allocate appropriate memory in one shot
4. Process in a single pass with the right accumulation method

This is much more efficient than the original two-pass approach and aligns perfectly with what we've already built for CPU optimization.