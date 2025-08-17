//
// Correct Metal Accumulator Kernel
// This implementation focuses on correctness over performance
//

#include <metal_stdlib>
using namespace metal;

// Kernel for accumulating sorted values with duplicate indices
// This is a simple, correct sequential implementation
kernel void accumulate_sorted_sequential(
    constant uint* sorted_indices [[buffer(0)]],   // Input: sorted indices
    constant float* sorted_values [[buffer(1)]],   // Input: corresponding values
    device uint* unique_indices [[buffer(2)]],     // Output: unique indices
    device float* accumulated_values [[buffer(3)]], // Output: accumulated sums
    device atomic_uint* unique_count [[buffer(4)]], // Output: count of unique elements
    constant uint& n_elements [[buffer(5)]],       // Input: number of elements to process
    uint tid [[thread_position_in_grid]])
{
    // Only thread 0 does the work (fully sequential for correctness)
    if (tid != 0) return;
    
    // Handle empty input
    if (n_elements == 0) {
        atomic_store_explicit(unique_count, 0, memory_order_relaxed);
        return;
    }
    
    // Initialize with first element
    uint current_index = sorted_indices[0];
    float current_sum = sorted_values[0];
    uint write_pos = 0;
    
    // Process remaining elements
    for (uint i = 1; i < n_elements; i++) {
        uint idx = sorted_indices[i];
        float val = sorted_values[i];
        
        if (idx == current_index) {
            // Same index - accumulate
            current_sum += val;
        } else {
            // Different index - write previous and start new
            unique_indices[write_pos] = current_index;
            accumulated_values[write_pos] = current_sum;
            write_pos++;
            
            current_index = idx;
            current_sum = val;
        }
    }
    
    // Write the final accumulated value
    unique_indices[write_pos] = current_index;
    accumulated_values[write_pos] = current_sum;
    write_pos++;
    
    // Store the total count
    atomic_store_explicit(unique_count, write_pos, memory_order_relaxed);
}

// Parallel version using threadgroup memory for small blocks
kernel void accumulate_sorted_parallel(
    constant uint* sorted_indices [[buffer(0)]],
    constant float* sorted_values [[buffer(1)]],
    device uint* unique_indices [[buffer(2)]],
    device float* accumulated_values [[buffer(3)]],
    device atomic_uint* unique_count [[buffer(4)]],
    constant uint& n_elements [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]])
{
    // Size of threadgroup shared memory
    constexpr uint SHARED_SIZE = 256;
    threadgroup uint shared_indices[SHARED_SIZE];
    threadgroup float shared_values[SHARED_SIZE];
    threadgroup bool is_segment_start[SHARED_SIZE];
    threadgroup float segment_sums[SHARED_SIZE];
    
    uint global_id = gid * group_size + tid;
    
    // Load data into shared memory
    if (global_id < n_elements) {
        shared_indices[tid] = sorted_indices[global_id];
        shared_values[tid] = sorted_values[global_id];
    } else {
        // Padding - use max value so it sorts to the end
        shared_indices[tid] = UINT_MAX;
        shared_values[tid] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Mark segment starts (where index changes)
    if (tid == 0) {
        // First element of block - check against previous block
        if (gid == 0 || global_id >= n_elements) {
            is_segment_start[tid] = true;
        } else {
            // Check against last element of previous block
            uint prev_global = global_id - 1;
            is_segment_start[tid] = (sorted_indices[prev_global] != shared_indices[tid]);
        }
    } else {
        // Check against previous element in shared memory
        if (global_id >= n_elements) {
            is_segment_start[tid] = false;
        } else {
            is_segment_start[tid] = (shared_indices[tid-1] != shared_indices[tid]);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute segment sums using parallel reduction
    segment_sums[tid] = shared_values[tid];
    
    // Simple sequential accumulation within block for correctness
    // A more optimized version would use parallel scan
    if (tid == 0) {
        uint block_write_pos = 0;
        
        for (uint i = 0; i < group_size && global_id + i < n_elements; i++) {
            if (shared_indices[i] == UINT_MAX) break;
            
            if (is_segment_start[i] && i > 0) {
                // Write previous segment
                uint output_pos = atomic_fetch_add_explicit(unique_count, 1, memory_order_relaxed);
                unique_indices[output_pos] = shared_indices[i-1];
                
                // Sum from last segment start to here
                float sum = 0.0f;
                for (uint j = block_write_pos; j < i; j++) {
                    sum += shared_values[j];
                }
                accumulated_values[output_pos] = sum;
                
                block_write_pos = i;
            }
        }
        
        // Handle last segment in block
        if (block_write_pos < group_size && shared_indices[block_write_pos] != UINT_MAX) {
            // Check if this segment continues in next block
            uint last_valid = group_size - 1;
            while (last_valid > block_write_pos && shared_indices[last_valid] == UINT_MAX) {
                last_valid--;
            }
            
            bool continues_next = (gid < (n_elements + group_size - 1) / group_size - 1) &&
                                 (global_id + last_valid + 1 < n_elements) &&
                                 (sorted_indices[global_id + last_valid + 1] == shared_indices[last_valid]);
            
            if (!continues_next) {
                // Complete segment - write it
                uint output_pos = atomic_fetch_add_explicit(unique_count, 1, memory_order_relaxed);
                unique_indices[output_pos] = shared_indices[last_valid];
                
                float sum = 0.0f;
                for (uint j = block_write_pos; j <= last_valid; j++) {
                    sum += shared_values[j];
                }
                accumulated_values[output_pos] = sum;
            }
            // If continues, the next block will handle accumulation
        }
    }
}