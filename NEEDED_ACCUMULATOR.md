# GPU Accumulator Specification for Sparse Matrix Operations

## Overview

We need a GPU kernel that takes sorted arrays of (index, value) pairs and accumulates (sums) all values that share the same index, producing a compressed output with unique indices and their accumulated sums.

## Problem Statement

In sparse matrix multiplication (SpGEMM), computing C = A × B produces intermediate products where multiple values may map to the same output position (i,j). These duplicates must be accumulated (summed) to produce the final result.

### Input
- `sorted_indices`: Array of unsigned 32-bit integers, sorted in ascending order
- `sorted_values`: Array of 32-bit floating-point values, corresponding to indices
- `n_elements`: Number of elements in the input arrays

### Output
- `unique_indices`: Array of unique indices from the input (sorted)
- `accumulated_values`: Array of accumulated sums for each unique index
- `unique_count`: Number of unique indices in the output

## Functional Requirements

### Core Functionality

1. **Accumulation Logic**:
   - For each unique index in the input, sum all corresponding values
   - Example: indices [0,0,1,1,1,2] with values [1,2,3,4,5,6] produces:
     - unique_indices: [0,1,2]
     - accumulated_values: [3,12,6]

2. **Correctness Requirements**:
   - Must handle arrays from size 1 to 10^7 elements
   - Must correctly handle cases where all indices are identical
   - Must correctly handle cases where all indices are unique
   - Must maintain numerical stability for floating-point accumulation

3. **Edge Cases**:
   - Empty input (n_elements = 0) → empty output
   - Single element → pass through unchanged
   - All duplicates → single output element with sum of all values
   - No duplicates → output identical to input

## Implementation Considerations

### Memory Management
- Output buffer sizing: Worst case requires n_elements entries (no duplicates)
- Actual output size is determined at runtime and returned via `unique_count`

### Performance Targets
- Should efficiently handle high-duplicate scenarios (>90% duplicates)
- Should efficiently handle low-duplicate scenarios (<10% duplicates)
- Target throughput: >1 GB/s for arrays >100K elements on modern GPUs

### Algorithm Options

#### Option 1: Sequential Scan (Simple, Correct)
```pseudocode
current_index = sorted_indices[0]
current_sum = sorted_values[0]
write_position = 0

for i = 1 to n_elements-1:
    if sorted_indices[i] == current_index:
        current_sum += sorted_values[i]
    else:
        unique_indices[write_position] = current_index
        accumulated_values[write_position] = current_sum
        write_position++
        current_index = sorted_indices[i]
        current_sum = sorted_values[i]

// Write final group
unique_indices[write_position] = current_index
accumulated_values[write_position] = current_sum
unique_count = write_position + 1
```

#### Option 2: Parallel Segmented Reduction
1. Mark segment boundaries (where indices change)
2. Perform parallel reduction within each segment
3. Compact results to output buffer

#### Option 3: Block-wise Processing
1. Divide input into blocks that fit in shared memory
2. Process each block independently
3. Merge results across blocks

## Test Cases

### Test 1: Basic Functionality
- Input: indices=[0,0,1,1,1,2,2,3,3,3,3], values=[1,2,3,4,5,6,7,8,9,10,11]
- Expected: unique=[0,1,2,3], accumulated=[3,12,13,38]

### Test 2: Large Array with Pattern
- Input: 50,000 elements where index = i % 1000, value = i * 0.1
- Expected: 1000 unique indices (0-999)
- Each index i should have sum = Σ(i + 1000k) * 0.1 for k=0..49

### Test 3: No Duplicates
- Input: indices=[0,1,2,3,4], values=[1,2,3,4,5]
- Expected: Same as input

### Test 4: All Duplicates
- Input: indices=[42,42,42,42], values=[1,2,3,4]
- Expected: unique=[42], accumulated=[10]

### Test 5: Powers of Two Boundary
- Test with sizes: 1023, 1024, 1025, 16383, 16384, 16385
- Verify correct handling regardless of array size

## Platform-Specific Notes

### Apple Metal
- Use atomic operations for accumulation if processing in parallel
- Handle threadgroup memory limits (typically 32KB)
- Account for Metal's threading model (threadgroups and threads)

### CUDA
- Use warp-level primitives for efficient reduction
- Consider using CUB library's segmented reduction

### Requirements for Correctness

1. **Deterministic Results**: Same input must always produce same output
2. **Floating-Point Accuracy**: Accumulation order may vary but results should be within IEEE-754 rounding tolerance
3. **No Data Races**: Parallel implementations must use appropriate synchronization
4. **Bounds Checking**: Never read/write outside allocated buffers
5. **Overflow Handling**: Gracefully handle index overflow (return error or clamp)

## Validation Criteria

An implementation is considered correct if:
1. All test cases pass with exact index matches
2. Accumulated values match within 1e-5 relative tolerance
3. No crashes or undefined behavior for any valid input size
4. Memory usage stays within 2x theoretical minimum
5. Performance scales appropriately with input size

## Reference

This accumulator is a critical component of the MAGNUS algorithm for sparse matrix multiplication (see https://arxiv.org/pdf/2501.07056). It handles the "accumulation" step after generating and sorting intermediate products during SpGEMM operations.