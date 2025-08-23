# MAGNUS Implementation Roadmap in Rust

MAGNUS is an algorithm described in [this paper](https://arxiv.org/pdf/2501.07056) for multiplying very large sparse matrices. Any uncertainties or ambiguities should be resolved by referring back to the source.

This roadmap outlines a comprehensive plan for building a high-performance implementation in Rust. The implementation will prioritize hardware-agnostic components first, followed by Intel-specific optimizations, and eventually ARM architecture support.

## Part 1: System Architecture and Technology Selection

### Language Selection: Rust

Rust is an excellent choice for this implementation because it provides:
- Zero-cost abstractions essential for high-performance computing
- Memory safety without garbage collection
- Excellent concurrency support via Rayon or similar libraries
- Direct access to SIMD intrinsics via `std::arch`
- Foreign function interface (FFI) for integrating with C/Assembly when needed

### Core Libraries and Dependencies

1. **Matrix Fundamentals**:
   - `sprs` - Rust sparse matrix library for basic formats and operations
   - `ndarray` - For dense array handling where needed

2. **Parallelism**:
   - `rayon` - Work-stealing parallel iterator library for data-parallel operations
   - Support for OpenMP semantics through `openmp-sys` bindings when needed

3. **SIMD/Vectorization**:
   - `std::arch` for direct SIMD intrinsics in Rust
   - For the AVX-512 sort, consider binding to:
     - Intel's x86-simd-sort library
     - Berenger Bramas' AVX-512 sorting implementation 

4. **Memory Management**:
   - `aligned-vec` - For properly aligned memory allocation (crucial for SIMD)
   - Custom allocators for non-temporal store operations

5. **Numeric Support**:
   - `num-traits` - For generic numeric operations

### Cross-Platform Architecture Design

To support both Intel (x86 with AVX-512) and ARM (Apple Silicon) architectures:

```rust
enum Architecture {
    X86WithAVX512,
    X86WithoutAVX512,
    ArmNeon,
    Generic,
}

// Architecture-aware configuration
struct MagnusConfig {
    // Core parameters
    system_params: SystemParameters,
    dense_accum_threshold: usize,
    sort_method: SortMethod,
    enable_coarse_level: bool,
    
    // Architecture-specific settings
    architecture: Architecture,
}
```

We'll implement a pluggable SIMD framework:

```rust
trait SimdAccelerator<T> {
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[T])
            -> (Vec<usize>, Vec<T>);
}

struct Avx512Accelerator;
struct NeonAccelerator;
struct FallbackAccelerator;
```

With conditional compilation for architecture-specific code:

```rust
#[cfg(target_arch = "x86_64")]
mod x86_implementation {
    // AVX-512 implementation
}

#[cfg(target_arch = "aarch64")]
mod arm_implementation {
    // ARM NEON implementation
}

// Feature detection at runtime
fn create_accelerator() -> Box<dyn SimdAccelerator<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_available() {
            return Box::new(Avx512Accelerator);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return Box::new(NeonAccelerator);
    }

    // Fallback for any architecture
    Box::new(FallbackAccelerator)
}
```

### Test Driven Development

We'll prioritize sections which can be tested on any architecture:

1. Before beginning work on a section:
   - Identify essential features
   - Write test coverage for each feature

2. Once work has been completed:
   - Test, reevaluate, rewrite as needed
   - Document mistakes in a separate document
   - Reassess roadmap if serious problems are encountered

3. For Intel-specific sections requiring AVX-512:
   - Work on Linux systems with appropriate hardware
   - Test Intel implementation first

4. After functionality tests are implemented:
   - Add performance tests for Intel hardware
   - Benchmark against the graphs in the original paper

## Part 2: Core Data Structures

### 1. Matrix Formats and Representation

```rust
// Base matrix types with CSR and CSC support
struct SparseMatrixCSR<T> {
    n_rows: usize,
    n_cols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
}

struct SparseMatrixCSC<T> {
    n_rows: usize,
    n_cols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<T>,
}

// For the coarse-level representation
struct AHatCSC<T> {
    original_row_indices: Vec<usize>,  // Maps to original rows in A
    matrix: SparseMatrixCSC<T>,
}
```

### 2. Intermediate Structures for Locality Generation

```rust
// Essential data structures for the reordering operations
struct ChunkMetadata {
    chunk_length: usize,
    n_chunks: usize,
    shift_bits: usize,  // For fast division by powers of 2
}

// For storing the temporary products during reordering
struct IntermediateProduct<T> {
    col_indices: Vec<usize>,
    values: Vec<T>,
}

// Fine-level reordering buffers
struct FineLevel<T> {
    counts: Vec<usize>,
    offsets: Vec<usize>,
    reordered_cols: Vec<usize>,
    reordered_vals: Vec<T>,
    metadata: ChunkMetadata,
}

// Coarse-level reordering buffers
struct CoarseLevel<T> {
    counts: Vec<Vec<usize>>,    // Per-row counts
    offsets: Vec<Vec<usize>>,   // Per-row offsets
    reordered_cols: Vec<usize>,
    reordered_vals: Vec<T>,
    metadata: ChunkMetadata,
}
```

### 3. Configuration and Parameters

```rust
struct SystemParameters {
    cache_line_size: usize,
    l2_cache_size: usize,
    n_threads: usize,
}

struct MagnusConfig {
    system_params: SystemParameters,
    dense_accum_threshold: usize,  // When to switch from sort to dense (default 256)
    sort_method: SortMethod,       // Pluggable sorting implementation
    enable_coarse_level: bool,     // Allow disabling for testing/comparison
    architecture: Architecture,    // Target architecture
}

enum SortMethod {
    SortThenReduce,
    ModifiedCompareExchange,
}

enum RowCategory {
    Sort,                // Small intermediate products
    DenseAccumulation,   // Fits in L2 cache
    FineLevel,           // Requires fine-level reordering
    CoarseLevel,         // Requires both coarse and fine-level reordering
}
```

## Part 3: Implementation Roadmap

### Stage 1: Foundation Layer (Weeks 1-2)

1. **Matrix Fundamentals**
   - Implement or adapt CSR/CSC matrix formats
   - Implement conversion between formats
   - Basic sparse matrix operations (for testing correctness)

2. **System Parameter Detection**
   - Cache size detection (using CPU feature detection)
   - Optimal parameter calculation (implementing Section 3.5 equations)

3. **Row Categorization Logic**
   - Implement the categorization logic for rows as described in Section 3.1
   - Determine which accumulation method to use for each row

```rust
fn categorize_rows<T>(
    a: &SparseMatrixCSR<T>, 
    b: &SparseMatrixCSR<T>, 
    config: &MagnusConfig,
) -> Vec<RowCategory> {
    // Analyze each row and categorize according to Section 3.1
    // 1. Small intermediate products -> Sort
    // 2. Intermediate product fits in L2 -> DenseAccumulation
    // 3. Fine-level storage fits in L2 -> FineLevel
    // 4. Else -> CoarseLevel
}
```

### Stage 2: Accumulators (Weeks 3-4)

1. **Dense Accumulator**
   - Implement the standard dense accumulator (Algorithm 1)
   - Optimize for different data types and cache efficiency

2. **Sort-Based Accumulator Framework**
   - Create pluggable interface for the sort implementation
   - Implement wrapper for threshold-based selection

3. **Hardware-Agnostic Sort-Then-Reduce Implementation**
   - Implement a generic version that works on all platforms
   - Design interface for later optimization with AVX-512/NEON

```rust
trait Accumulator<T> {
    fn accumulate(&self, col_indices: &[usize], values: &[T]) -> (Vec<usize>, Vec<T>);
}

struct DenseAccumulator<T> {
    max_col_idx: usize,
}

struct SortAccumulator<T> {
    method: SortMethod,
    accelerator: Box<dyn SimdAccelerator<T>>,
}

// Generic implementation for all platforms
fn sort_then_reduce<T: AddAssign + Copy>(
    col_indices: &[usize], 
    values: &[T]
) -> (Vec<usize>, Vec<T>) {
    // 1. Sort pairs
    // 2. Scan for duplicates and accumulate
}
```

### Stage 3: Fine-Level Algorithm (Weeks 5-6)

1. **Histogram and Prefix Sum**
   - Implement the histogram computation (Algorithm 3, lines 2-6)
   - Implement prefix sum with platform-agnostic optimization

2. **Reordering Logic**
   - Implement the reordering step (Algorithm 3, lines 11-17)
   - Hardware-agnostic optimizations for memory access patterns

3. **Fine-Level Integration**
   - Connect the histogram, prefix sum, reorder, and accumulate steps
   - Implement chunk-based accumulation logic

```rust
fn fine_level_histogram(col_indices: &[usize], metadata: &ChunkMetadata) -> Vec<usize> {
    let mut counts = vec![0; metadata.n_chunks];
    for &col in col_indices {
        let chunk = col >> metadata.shift_bits;
        counts[chunk] += 1;
    }
    counts
}

fn fine_level_reorder<T: Copy>(
    col_indices: &[usize], 
    values: &[T],
    offsets: &[usize],
    metadata: &ChunkMetadata,
) -> (Vec<usize>, Vec<T>) {
    // Implement memory-efficient reordering
    // Use the bitshift-based mapping for performance
}

fn process_fine_level<T: AddAssign + Copy>(
    col_indices: &[usize],
    values: &[T],
    config: &MagnusConfig,
) -> (Vec<usize>, Vec<T>) {
    // Full implementation of Algorithm 3
}
```

### Stage 4: Coarse-Level Algorithm (Weeks 7-9)

1. **AˆCSC Construction (Gap 1)**
   - Implement subset extraction and CSR to CSC conversion
   - Optimize for parallel execution

2. **Batching Strategy (Gap 3)**
   - Implement heuristic for coarse batch size determination
   - Memory-aware batching with configurable limits

3. **Coarse-Level Reordering**
   - Implement the histogram, prefix sum, and reordering steps
   - Optimize memory access patterns

4. **Integration with Fine-Level**
   - Apply fine-level to each coarse-level chunk
   - Merge results into final output

```rust
// Gap 1: AˆCSC Construction
fn construct_a_hat_csc<T: Copy>(
    a: &SparseMatrixCSR<T>,
    coarse_rows_c: &[usize],
) -> AHatCSC<T> {
    // Two-pass algorithm:
    // 1. Count nonzeros per column
    // 2. Populate CSC data structure
}

// Gap 3: Coarse batch size determination
fn determine_coarse_batch_size(
    a: &SparseMatrixCSR<f32>,
    config: &MagnusConfig,
) -> usize {
    // Implement the heuristic from Section 3.3
    // Consider both memory limits and L2 cache size
}

fn process_coarse_level<T: AddAssign + Copy>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    coarse_rows_c: &[usize],
    config: &MagnusConfig,
) -> SparseMatrixCSR<T> {
    // Full implementation of Algorithm 4
}
```

### Stage 5: Architecture-Specific Optimizations (Weeks 10-12)

1. **Intel AVX-512 Optimizations**
   - Implement AVX-512 sort-then-reduce
   - Optimize for Intel hardware

2. **Modified Compare-Exchange (Gap 4 Alternative)**
   - Research existing implementations in Fortran/MATLAB
   - Implement AVX-512 bitonic sort with integrated accumulation

3. **ARM NEON Implementation (Future)**
   - Scaffolding for ARM NEON vectorized sorting
   - Performance parameter retuning for Apple Silicon

```rust
// Intel-specific implementation
#[cfg(target_arch = "x86_64")]
impl<T: AddAssign + Copy> SimdAccelerator<T> for Avx512Accelerator {
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[T])
            -> (Vec<usize>, Vec<T>) {
        // AVX-512 implementation
    }
}

// Gap 4: Modified Compare-Exchange implementation for Intel
#[cfg(target_arch = "x86_64")]
fn bitonic_sort_with_accumulation<T: AddAssign + Copy>(
    col_indices: &[usize],
    values: &[T],
) -> (Vec<usize>, Vec<T>) {
    // Implement modified compare-exchange operations
    // that detect equality and accumulate values
}

// ARM-specific implementation (for future development)
#[cfg(target_arch = "aarch64")]
impl<T: AddAssign + Copy> SimdAccelerator<T> for NeonAccelerator {
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[T])
            -> (Vec<usize>, Vec<T>) {
        // ARM NEON implementation
    }
}
```

### Stage 6: Integration and Main Algorithm (Weeks 13-14)

1. **SpGEMM Driver**
   - Implement the main SpGEMM function
   - Set up the workflow for symbolic and numeric phases

2. **Row Processing by Category**
   - Process rows according to their category
   - Group rows by category for better cache efficiency

3. **Parallel Execution Strategy**
   - Implement work distribution across threads
   - Handle synchronization points appropriately

```rust
pub fn magnus_spgemm<T: AddAssign + Copy + Num>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    config: &MagnusConfig,
) -> SparseMatrixCSR<T> {
    // 1. Setup phase (parameters, categorization)
    // 2. Symbolic phase
    // 3. Numeric phase
    //    - Process rows by category
    //    - Apply appropriate algorithm for each category
}
```

### Stage 7: Performance Optimization (Weeks 15-16)

1. **Benchmarking Infrastructure**
   - Implement test matrices from the paper
   - Set up performance measurement framework

2. **Microbenchmarks for Parameter Tuning**
   - Benchmark accumulator threshold selection
   - Tune coarse/fine level parameters

3. **Profiling and Hotspot Optimization**
   - Identify and optimize performance bottlenecks
   - Fine-tune SIMD utilization

```rust
fn benchmark_accumulator_crossover(
    max_size: usize,
    step: usize,
) -> usize {
    // Similar to Figure 4 in the paper
    // Determine optimal threshold for switching between sort and dense
}
```

## Part 4: Addressing the Implementation Gaps

### Gap 1: Coarse-Level Algorithm Details

```rust
// Detailed implementation for AˆCSC construction
fn construct_a_hat_csc<T: Copy>(
    a: &SparseMatrixCSR<T>,
    coarse_rows_c: &[usize],
) -> AHatCSC<T> {
    // Create bitmap to track which rows are needed
    let mut row_bitmap = vec![false; a.n_rows];
    for &row in coarse_rows_c {
        row_bitmap[row] = true;
    }
    
    // Count nonzeros per column (parallel)
    let mut col_counts = vec![0; a.n_cols];
    
    // Parallel version using Rayon
    coarse_rows_c.par_iter().for_each(|&row| {
        let row_start = a.row_ptr[row];
        let row_end = a.row_ptr[row + 1];
        
        for idx in row_start..row_end {
            let col = a.col_idx[idx];
            // Use atomic increment for thread safety
            atomic_increment(&mut col_counts[col]);
        }
    });
    
    // Compute column pointers via prefix sum
    let col_ptr = exclusive_scan(&col_counts);
    
    // Allocate arrays for CSC matrix
    let nnz = col_ptr[a.n_cols];
    let mut row_idx = vec![0; nnz];
    let mut values = vec![T::zero(); nnz];
    
    // Fill CSC matrix (second pass)
    let mut write_pos = col_ptr.clone();
    
    for row in 0..a.n_rows {
        if !row_bitmap[row] {
            continue;
        }
        
        let row_start = a.row_ptr[row];
        let row_end = a.row_ptr[row + 1];
        
        for idx in row_start..row_end {
            let col = a.col_idx[idx];
            let pos = write_pos[col];
            
            row_idx[pos] = row;
            values[pos] = a.values[idx];
            write_pos[col] += 1;
        }
    }
    
    AHatCSC {
        original_row_indices: coarse_rows_c.to_vec(),
        matrix: SparseMatrixCSC {
            n_rows: a.n_rows,
            n_cols: a.n_cols,
            col_ptr,
            row_idx,
            values,
        }
    }
}
```

### Gap 2: Accumulator Switch Threshold

```rust
// Microbenchmark to determine the optimal threshold
fn determine_optimal_threshold(config: &MagnusConfig) -> usize {
    // Default to 256 based on paper
    let default_threshold = 256;
    
    if config.auto_tune {
        // Run microbenchmarks similar to Figure 4
        // Test both approaches on varying input sizes
        // Return the crossover point
        benchmark_accumulator_crossover(512, 8)
    } else {
        default_threshold
    }
}
```

### Gap 3: Coarse-Level Batch Size

```rust
fn determine_coarse_batch_size(
    a: &SparseMatrixCSR<f32>,
    b: &SparseMatrixCSR<f32>,
    rows_requiring_coarse: &[usize],
    config: &MagnusConfig,
) -> usize {
    // Implement the heuristic based on memory constraints
    // Memory limit condition
    let estimated_memory_per_row = estimate_memory_requirement_per_row(a, b, rows_requiring_coarse);
    let memory_limited_batch_size = config.available_memory / estimated_memory_per_row;
    
    // L2 cache limit condition for coarse-level data structures
    let coarse_structures_size = calculate_coarse_structures_size(config);
    let cache_limited_batch_size = config.system_params.l2_cache_size / coarse_structures_size;
    
    // Take the minimum of the two limits
    min(memory_limited_batch_size, cache_limited_batch_size)
}
```

### Gap 4: AVX-512 Sort with Accumulation

We'll implement both approaches with a pluggable interface:

```rust
// Sort-then-reduce implementation
fn avx512_sort_then_reduce<T: AddAssign + Copy>(
    col_indices: &[usize],
    values: &[T],
) -> (Vec<usize>, Vec<T>) {
    // 1. Sort using Intel's x86-simd-sort or similar
    let (sorted_indices, sorted_values) = avx512_sort_key_value(col_indices, values);
    
    // 2. Scan for duplicates and accumulate
    let mut result_indices = Vec::new();
    let mut result_values = Vec::new();
    
    if sorted_indices.is_empty() {
        return (result_indices, result_values);
    }
    
    let mut current_idx = sorted_indices[0];
    let mut current_val = sorted_values[0];
    
    for i in 1..sorted_indices.len() {
        if sorted_indices[i] == current_idx {
            // Accumulate duplicate
            current_val += sorted_values[i];
        } else {
            // Output accumulated value and start new group
            result_indices.push(current_idx);
            result_values.push(current_val);
            
            current_idx = sorted_indices[i];
            current_val = sorted_values[i];
        }
    }
    
    // Don't forget the last group
    result_indices.push(current_idx);
    result_values.push(current_val);
    
    (result_indices, result_values)
}

// Modified compare-exchange implementation 
// We'll need to implement or find a bitonic sorting network that
// handles accumulation during the compare-exchange steps
fn avx512_bitonic_sort_with_accumulation<T: AddAssign + Copy>(
    col_indices: &[usize],
    values: &[T],
) -> (Vec<usize>, Vec<T>) {
    // This would be a complex implementation using AVX-512 intrinsics
    // If we find existing implementations from Fortran/MATLAB,
    // we should consider binding to them
    
    // The implementation would need to:
    // 1. Detect equality during compare-exchange steps
    // 2. Accumulate values for equal keys
    // 3. Mark or handle duplicate keys appropriately
    
    // For now, we'll use the sort-then-reduce approach as a fallback
    avx512_sort_then_reduce(col_indices, values)
}
```

## Part 5: Testing and Benchmarking Strategy

### 1. Correctness Testing

```rust
fn test_against_reference_implementation<T: PartialEq + AddAssign + Copy>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
) {
    // Compute with our implementation
    let c_magnus = magnus_spgemm(a, b, &default_config());
    
    // Compute with a reference implementation (e.g., sprs)
    let c_reference = reference_spgemm(a, b);
    
    // Compare results
    assert_eq!(c_magnus, c_reference);
}
```

### 2. Performance Benchmarking

```rust
fn benchmark_matrix_suite(suite_name: &str, matrices: &[TestMatrix]) -> BenchmarkResults {
    let configs = vec![
        MagnusConfig { enable_coarse_level: false, ..default_config() }, // Fine-level only
        default_config(),                                                // Full MAGNUS
    ];
    
    // Also benchmark baseline implementations
    let baseline_implementations = vec![
        "CSeg",
        "MKL",
        "Hash",
        "Heap",
        "GraphBLAS",
    ];
    
    // Run all combinations and collect results
    // Plot similar to Figures 7-9 in the paper
}
```

### 3. Synthetic Matrix Generation

```rust
fn generate_test_matrices() -> Vec<TestMatrix> {
    let mut matrices = Vec::new();
    
    // SuiteSparse matrices from Table 2
    matrices.extend(load_suitesparse_matrices());
    
    // R-mat matrices from Table 3
    for scale in 18..=23 {
        matrices.push(generate_rmat(scale, 16.0)); // 16 nonzeros per row
    }
    
    // Uniform random matrices (varying columns)
    for cols_exp in 20..=35 {
        let cols = 1 << cols_exp;
        matrices.push(generate_uniform_random(4096, cols, 2048));
    }
    
    matrices
}
```

## Part 6: Cross-Platform Performance Expectations

1. **Intel with AVX-512**: Will likely achieve performance closest to the paper's results.

2. **Other x86_64 Platforms**: Will work with the generic implementation, with performance varying based on available SIMD instructions.

3. **Apple Silicon**: Will use ARM NEON optimizations but probably won't match AVX-512 performance for the sorting accelerator. The overall algorithm's locality generation benefits would still apply.

4. **Generic Fallback**: The implementation would still work on all platforms but with varying performance levels.

## Conclusion

This roadmap provides a structured approach to implementing the MAGNUS algorithm in Rust with cross-platform support. The implementation strategy:

1. Prioritizes hardware-agnostic components first
2. Uses Rust's safety features while maintaining performance
3. Leverages existing libraries for core functionality
4. Addresses each implementation gap with practical solutions
5. Provides a pluggable approach to architecture-specific optimizations
6. Includes comprehensive testing and benchmarking

The most challenging aspect will be the AVX-512 sort with accumulation (Gap 4), but the proposed sort-then-reduce approach offers a practical starting point while we investigate more optimized solutions. For truly performance-critical sections, we can fall back to C or even assembly through Rust's FFI capabilities if necessary.