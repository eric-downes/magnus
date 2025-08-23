# Constants Refactoring Summary

## Overview

All hardcoded numeric constants have been extracted from the MAGNUS codebase and centralized in `src/constants.rs`. This refactoring was undertaken to prevent bugs like the bitonic sort issue and improve maintainability.

## Changes Made

### 1. Created `src/constants.rs`

A new file containing all numeric constants organized into categories:
- Architecture-specific constants (vector widths, chunk sizes)
- Accumulator thresholds
- SIMD processing thresholds
- Memory and cache constants
- Prefetch strategy constants
- Matrix density and sparsity thresholds
- Floating point tolerances
- Display and debug constants
- Benchmarking constants
- Test matrix generation constants
- Metal GPU kernel constants
- Bitonic sort constants

### 2. Updated Implementation Files

The following files were updated to use centralized constants:

#### Core Implementation Files
- `src/accumulator/mod.rs` - Initial capacity constants
- `src/accumulator/metal_impl.rs` - GPU thresholds and padding values
- `src/accumulator/accelerate.rs` - SIMD thresholds
- `src/matrix/config.rs` - Architecture-specific parameters
- `src/matrix/csr.rs` - Display constants
- `src/matrix/csc.rs` - Display constants
- `src/matrix/spgemm_prefetch.rs` - Density and NNZ thresholds
- `src/utils/prefetch.rs` - Memory thresholds and hit rates

### 3. Documentation Updates

#### Updated Files
- `docs/roadmap.md` - Added "Code Standards and Best Practices" section
- `CONTRIBUTING.md` - Added constants management as critical requirement
- `README.md` - Added note about constants requirement

#### Key Guidelines Established
1. Never hardcode numeric literals in implementation code
2. Always define constants in `src/constants.rs`
3. Use descriptive ALL_CAPS names
4. Document each constant with a comment
5. Group related constants under section headers

## Constants Extracted

### Key Constants (Examples)

| Constant | Value | Purpose |
|----------|-------|---------|
| `METAL_GPU_THRESHOLD` | 10,000 | Minimum elements for GPU acceleration |
| `AVX512_VECTOR_WIDTH_BYTES` | 64 | Vector width for AVX-512 |
| `NEON_VECTOR_WIDTH_BYTES` | 16 | Vector width for ARM NEON |
| `SPARSE_DENSITY_THRESHOLD` | 0.001 | Threshold for sparse matrix classification |
| `DEFAULT_L2_CACHE_SIZE` | 256KB | Default L2 cache size |
| `HIGH_HIT_RATE_THRESHOLD` | 0.9 | High cache hit rate threshold |
| `FLOAT_COMPARISON_EPSILON` | 1e-10 | Standard floating point tolerance |
| `MAX_DISPLAY_ROWS` | 5 | Maximum rows in debug display |
| `BITONIC_SORT_PADDING_INDEX` | u32::MAX | Padding value for bitonic sort |

### Total Constants Defined
- 60+ named constants extracted and centralized
- Organized into 12 categories
- All with documentation comments

## Benefits Achieved

1. **Bug Prevention**: Eliminates magic number bugs like the bitonic sort issue
2. **Maintainability**: Easy to tune algorithm parameters in one place
3. **Documentation**: Clear record of design decisions and thresholds
4. **Consistency**: Ensures uniform behavior across the codebase
5. **Testing**: Easier to test edge cases and boundary conditions
6. **Performance Tuning**: Centralized location for optimization parameters

## Testing

All tests pass after refactoring:
- Unit tests: ✅
- Integration tests: ✅
- Metal bitonic sort tests: ✅ (12 passed, 0 failed)
- Build: ✅ Successful with warnings only

## Future Maintenance

Going forward, all contributors must:
1. Check `src/constants.rs` before adding any numeric literal
2. Add new constants to appropriate category
3. Document the purpose of each constant
4. Never hardcode magic numbers in implementation

This refactoring establishes a solid foundation for preventing constant-related bugs and improving code maintainability.