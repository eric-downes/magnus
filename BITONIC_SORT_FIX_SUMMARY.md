# Bitonic Sort Fix Summary

## The Bug

The original bitonic sort kernel had an incorrect direction calculation:
```metal
bool ascending = ((tid & stage) == 0);  // WRONG
```

This made the sort direction dependent on individual thread IDs rather than blocks of threads, violating the fundamental invariant of bitonic sort that all comparisons within a bitonic sequence must use the same direction.

## The Fix

The corrected version:
```metal
uint block_id = tid / stage;
bool ascending = (block_id & 2) == 0;
```

This ensures threads are grouped into blocks based on the stage size, and blocks alternate between ascending and descending sort directions.

## Additional Issues Fixed

1. **Padding Handling**: Fixed the accumulation logic to properly handle padded values (u32::MAX) that are added to make the array size a power of two.

2. **Buffer Reading**: Changed to read the full padded buffer size to ensure all sorted elements are captured.

3. **Accumulation Logic**: Improved the CPU-side accumulation to correctly skip padding values and handle edge cases.

## Structural Type Design

Created `bitonic_sort_types.rs` module with types that enforce invariants structurally:

### Key Types

1. **PowerOfTwo**: Guarantees a value is a power of two at the type level
   - Stores both the value and its log2
   - Provides safe constructors and conversions

2. **BitonicStage**: Represents a stage in the bitonic sort
   - Encapsulates stage number and size
   - Provides methods for number of passes

3. **BitonicPass**: Represents a pass within a stage
   - Tracks the comparison distance
   - Calculates block size for direction determination

4. **BitonicThread**: Thread context for bitonic sort
   - Encapsulates thread ID
   - Provides methods to find partner thread
   - Determines sort direction structurally
   - Checks if comparison should be performed

5. **BitonicSortConfig**: Overall configuration
   - Manages padding size calculation
   - Provides iterators over stages and passes
   - Encapsulates the algorithm structure

### Benefits of Structural Types

1. **Clarity**: The algorithm's structure is explicit in the type system
2. **Safety**: Invalid states (like non-power-of-two sizes) are impossible
3. **Documentation**: Types serve as documentation of invariants
4. **Debugging**: Easier to reason about and test individual components
5. **No Magic**: Eliminates clever bit tricks in favor of clear semantics

### Example Usage

```rust
let config = BitonicSortConfig::new(n_elements);

for stage in config.stages() {
    for pass in config.passes(stage) {
        // Each thread determines its behavior structurally
        let thread = BitonicThread::new(thread_id);
        if thread.should_compare(&pass, n_elements) {
            let partner = thread.partner(&pass);
            let ascending = thread.is_ascending(&pass);
            // Perform comparison...
        }
    }
}
```

This design makes the bitonic sort algorithm's structure explicit and eliminates the possibility of the direction calculation bug through type safety.

## Testing

Added comprehensive tests in `tests/metal_bitonic_sort.rs` covering:
- Empty input
- Single element
- Already sorted data
- Reverse sorted data
- All same index (maximum accumulation)
- Random patterns with duplicates
- Power-of-two and non-power-of-two sizes
- Sparse matrix patterns
- Edge values (near u32::MAX)
- Numerical stability with small values
- Large scale stress tests

All tests now pass, confirming the fix is correct.