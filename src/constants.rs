//! Centralized constants for the MAGNUS sparse matrix multiplication library
//!
//! This module contains all hardcoded constants used throughout the codebase.
//! All new constants should be added here rather than scattered throughout the code.
//! Constants are organized by category for easy reference and maintenance.

// ============================================================================
// ARCHITECTURE-SPECIFIC CONSTANTS
// ============================================================================

/// Vector width in bytes for AVX-512 architecture
pub const AVX512_VECTOR_WIDTH_BYTES: usize = 64;

/// Vector width in bytes for AVX2 architecture
pub const AVX2_VECTOR_WIDTH_BYTES: usize = 32;

/// Vector width in bytes for ARM NEON architecture
pub const NEON_VECTOR_WIDTH_BYTES: usize = 16;

/// Vector width in bytes for generic/scalar processing
pub const SCALAR_VECTOR_WIDTH_BYTES: usize = 8;

/// Optimal chunk size for AVX-512 processing
pub const AVX512_OPTIMAL_CHUNK_SIZE: usize = 2048;

/// Optimal chunk size for AVX2 processing
pub const AVX2_OPTIMAL_CHUNK_SIZE: usize = 1024;

/// Optimal chunk size for ARM NEON processing
pub const NEON_OPTIMAL_CHUNK_SIZE: usize = 512;

/// Optimal chunk size for generic processing
pub const GENERIC_OPTIMAL_CHUNK_SIZE: usize = 256;

// ============================================================================
// ACCUMULATOR THRESHOLDS
// ============================================================================

/// Dense accumulator threshold for AVX-512
pub const AVX512_DENSE_THRESHOLD: usize = 256;

/// Dense accumulator threshold for AVX2
pub const AVX2_DENSE_THRESHOLD: usize = 256;

/// Dense accumulator threshold for ARM NEON
pub const NEON_DENSE_THRESHOLD: usize = 192;

/// Dense accumulator threshold for generic processing
pub const GENERIC_DENSE_THRESHOLD: usize = 256;

/// Threshold for using Metal GPU acceleration (number of elements)
pub const METAL_GPU_THRESHOLD: usize = 10_000;

/// Size threshold for GPU processing decisions
pub const METAL_SIZE_THRESHOLD: usize = 32;

/// Divisor for initial capacity calculation in accumulators
pub const INITIAL_CAPACITY_DIVISOR: usize = 10;

/// Maximum initial capacity for sort accumulator
pub const MAX_SORT_ACCUMULATOR_CAPACITY: usize = 1024;

/// Default initial capacity for sort accumulator
pub const DEFAULT_SORT_ACCUMULATOR_SIZE: usize = 256;

// ============================================================================
// SIMD PROCESSING THRESHOLDS
// ============================================================================

/// Minimum elements for AVX-512 SIMD processing
pub const AVX512_MIN_ELEMENTS: usize = 32;

/// Threshold for using scalar fallback in Accelerate framework
pub const ACCELERATE_SIMD_THRESHOLD: usize = 32;

/// Minimum elements for NEON SIMD processing
pub const NEON_MIN_ELEMENTS: usize = 4;

// ============================================================================
// MEMORY AND CACHE CONSTANTS
// ============================================================================

/// Common cache line size in bytes
pub const DEFAULT_CACHE_LINE_SIZE: usize = 64;

/// Default L2 cache size (256KB)
pub const DEFAULT_L2_CACHE_SIZE: usize = 256_000;

/// L2 cache size for testing (256KB)
pub const TEST_L2_CACHE_SIZE: usize = 256 * 1024;

/// Conservative memory threshold (2GB)
pub const CONSERVATIVE_MEMORY_THRESHOLD: usize = 2 * 1024 * 1024 * 1024;

/// Moderate memory threshold (4GB)
pub const MODERATE_MEMORY_THRESHOLD: usize = 4 * 1024 * 1024 * 1024;

/// Aggressive memory threshold (8GB)
pub const AGGRESSIVE_MEMORY_THRESHOLD: usize = 8 * 1024 * 1024 * 1024;

/// Low memory detection threshold (4GB)
pub const LOW_MEMORY_DETECTION_THRESHOLD: usize = 4 * 1024 * 1024 * 1024;

/// Medium memory detection threshold (8GB)
pub const MEDIUM_MEMORY_DETECTION_THRESHOLD: usize = 8 * 1024 * 1024 * 1024;

// ============================================================================
// PREFETCH STRATEGY CONSTANTS
// ============================================================================

/// Distance threshold for prefetch strategy decisions
pub const PREFETCH_DISTANCE_THRESHOLD: usize = 2;

/// Multiplier for prefetch count calculation
pub const PREFETCH_COUNT_MULTIPLIER: usize = 2;

/// High hit rate threshold for prefetch strategy
pub const HIGH_HIT_RATE_THRESHOLD: f64 = 0.9;

/// Medium hit rate threshold for prefetch strategy
pub const MEDIUM_HIT_RATE_THRESHOLD: f64 = 0.7;

// ============================================================================
// MATRIX DENSITY AND SPARSITY THRESHOLDS
// ============================================================================

/// Density threshold for considering a matrix sparse
pub const SPARSE_DENSITY_THRESHOLD: f64 = 0.001;

/// Average NNZ threshold for B matrix processing decisions
pub const B_MATRIX_AVG_NNZ_THRESHOLD: usize = 100;

/// Minimum average NNZ threshold for A and B matrices
pub const MIN_AVG_NNZ_THRESHOLD: usize = 10;

// ============================================================================
// FLOATING POINT TOLERANCES
// ============================================================================

/// Standard floating point comparison epsilon
pub const FLOAT_COMPARISON_EPSILON: f64 = 1e-10;

/// Tolerance for Metal GPU computations
pub const METAL_COMPUTATION_TOLERANCE: f32 = 1e-5;

/// Relative error tolerance for large computations
pub const LARGE_COMPUTATION_RELATIVE_ERROR: f32 = 1e-3;

// ============================================================================
// DISPLAY AND DEBUG CONSTANTS
// ============================================================================

/// Maximum rows to print in debug display
pub const MAX_DISPLAY_ROWS: usize = 5;

/// Maximum elements per row in debug display
pub const MAX_DISPLAY_ELEMENTS_PER_ROW: usize = 5;

// ============================================================================
// CATEGORIZATION CONSTANTS
// ============================================================================

/// Default chunk log for fine-level structures
pub const DEFAULT_CHUNK_LOG: usize = 2;

/// Percentage conversion factor
pub const PERCENTAGE_CONVERSION_FACTOR: f64 = 100.0;

// ============================================================================
// BENCHMARKING CONSTANTS
// ============================================================================

/// Minimum NNZ variation factor for benchmarks
pub const MIN_NNZ_FACTOR: f64 = 0.5; // 1/2

/// Maximum NNZ variation factor for benchmarks
pub const MAX_NNZ_FACTOR: f64 = 1.5; // 3/2

/// Maximum attempts for generating unique columns in benchmarks
pub const MAX_COL_GENERATION_ATTEMPTS: usize = 100;

/// Small benchmark measurement time in seconds
pub const SMALL_BENCH_TIME_SECS: u64 = 5;

/// Small benchmark sample size
pub const SMALL_BENCH_SAMPLES: usize = 10;

/// Large benchmark measurement time in seconds
pub const LARGE_BENCH_TIME_SECS: u64 = 30;

/// Large benchmark sample size
pub const LARGE_BENCH_SAMPLES: usize = 10;

// ============================================================================
// TEST MATRIX GENERATION CONSTANTS
// ============================================================================

/// Ultra-sparse matrix density for testing
pub const ULTRA_SPARSE_DENSITY: f64 = 0.001;

/// Sparse matrix density for testing
pub const SPARSE_DENSITY: f64 = 0.01;

/// Medium matrix density for testing
pub const MEDIUM_DENSITY: f64 = 0.1;

/// Dense matrix density for testing
pub const DENSE_DENSITY: f64 = 0.2;

/// Small matrix size for testing
pub const SMALL_MATRIX_SIZE: usize = 1000;

/// Medium matrix size for testing
pub const MEDIUM_MATRIX_SIZE: usize = 5000;

/// Large matrix size for testing
pub const LARGE_MATRIX_SIZE: usize = 10000;

/// Extra large matrix size for testing
pub const XLARGE_MATRIX_SIZE: usize = 100000;

// ============================================================================
// METAL GPU KERNEL CONSTANTS
// ============================================================================

/// Maximum threads per threadgroup for Metal kernels
pub const METAL_MAX_THREADS_PER_GROUP: usize = 256;

/// Default threads per threadgroup for Metal kernels
pub const METAL_DEFAULT_THREADS_PER_GROUP: usize = 256;

// ============================================================================
// BITONIC SORT CONSTANTS
// ============================================================================

/// Sentinel value for padding in bitonic sort (u32::MAX)
pub const BITONIC_SORT_PADDING_INDEX: u32 = u32::MAX;

/// Sentinel value for padding in bitonic sort (f32::INFINITY)
pub const BITONIC_SORT_PADDING_VALUE: f32 = f32::INFINITY;
