//! Parameter space exploration framework for MAGNUS
//!
//! This module defines parameter groupings and joint distributions for
//! systematic testing across the algorithm's configuration space.

use crate::constants::*;
use crate::matrix::{SparseMatrixCSR, SparseMatrixCSC};
use rand::distributions::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Number of random objects to generate per parameter combination
pub const NUM_RAND_OBJS: usize = 3;

// ============================================================================
// PARAMETER GROUPINGS
// ============================================================================

/// Group 1: Matrix Structure Parameters
/// These parameters define the fundamental structure of test matrices
#[derive(Debug, Clone)]
pub struct MatrixStructureParams {
    pub size: usize,
    pub density: f64,
    pub avg_nnz_per_row: usize,
}

/// Group 2: Algorithm Selection Parameters
/// These thresholds determine which algorithm path is taken
#[derive(Debug, Clone)]
pub struct AlgorithmSelectionParams {
    pub dense_threshold: usize,
    pub gpu_threshold: usize,
    pub simd_min_elements: usize,
}

/// Group 3: Memory Hierarchy Parameters
/// These control cache and memory behavior
#[derive(Debug, Clone)]
pub struct MemoryHierarchyParams {
    pub cache_line_size: usize,
    pub l2_cache_size: usize,
    pub chunk_size: usize,
    pub memory_threshold: usize,
}

/// Group 4: Prefetch Strategy Parameters
/// These control prefetching behavior
#[derive(Debug, Clone)]
pub struct PrefetchParams {
    pub distance_threshold: usize,
    pub count_multiplier: usize,
    pub hit_rate_threshold: f64,
}

/// Group 5: Accumulator Configuration
/// These control accumulator behavior and initialization
#[derive(Debug, Clone)]
pub struct AccumulatorParams {
    pub initial_capacity_divisor: usize,
    pub max_capacity: usize,
    pub default_size: usize,
}

// ============================================================================
// PARAMETER RANGES AND DISTRIBUTIONS
// ============================================================================

/// Defines the exploration space for matrix structure
#[derive(Clone)]
pub struct MatrixStructureSpace {
    /// Size categories: small, medium, large, xlarge
    pub size_points: Vec<usize>,
    /// Density categories: ultra-sparse, sparse, medium, dense
    pub density_points: Vec<f64>,
    /// NNZ distribution: low, medium, high
    pub nnz_ranges: Vec<(usize, usize)>,
}

impl Default for MatrixStructureSpace {
    fn default() -> Self {
        Self {
            size_points: vec![
                SMALL_MATRIX_SIZE,
                MEDIUM_MATRIX_SIZE,
                LARGE_MATRIX_SIZE,
                XLARGE_MATRIX_SIZE,
            ],
            density_points: vec![
                ULTRA_SPARSE_DENSITY,
                SPARSE_DENSITY,
                MEDIUM_DENSITY,
                DENSE_DENSITY,
            ],
            nnz_ranges: vec![
                (1, MIN_AVG_NNZ_THRESHOLD),
                (MIN_AVG_NNZ_THRESHOLD, B_MATRIX_AVG_NNZ_THRESHOLD),
                (B_MATRIX_AVG_NNZ_THRESHOLD, 1000),
            ],
        }
    }
}

/// Defines the exploration space for algorithm selection
#[derive(Clone)]
pub struct AlgorithmSelectionSpace {
    /// Dense threshold variations around architecture defaults
    pub dense_threshold_factors: Vec<f64>,
    /// GPU threshold variations
    pub gpu_threshold_factors: Vec<f64>,
    /// SIMD threshold variations
    pub simd_threshold_factors: Vec<f64>,
}

impl Default for AlgorithmSelectionSpace {
    fn default() -> Self {
        Self {
            dense_threshold_factors: vec![0.5, 1.0, 2.0],
            gpu_threshold_factors: vec![0.1, 1.0, 10.0],
            simd_threshold_factors: vec![0.5, 1.0, 2.0],
        }
    }
}

/// Defines the exploration space for memory hierarchy
#[derive(Clone)]
pub struct MemoryHierarchySpace {
    /// L2 cache size variations
    pub l2_cache_factors: Vec<f64>,
    /// Chunk size variations relative to optimal
    pub chunk_size_factors: Vec<f64>,
    /// Memory threshold levels
    pub memory_levels: Vec<usize>,
}

impl Default for MemoryHierarchySpace {
    fn default() -> Self {
        Self {
            l2_cache_factors: vec![0.5, 1.0, 2.0, 4.0],
            chunk_size_factors: vec![0.25, 0.5, 1.0, 2.0],
            memory_levels: vec![
                CONSERVATIVE_MEMORY_THRESHOLD,
                MODERATE_MEMORY_THRESHOLD,
                AGGRESSIVE_MEMORY_THRESHOLD,
            ],
        }
    }
}

// ============================================================================
// PARAMETER COMBINATIONS
// ============================================================================

/// Represents a complete parameter configuration
#[derive(Debug, Clone)]
pub struct ParameterConfiguration {
    pub matrix: MatrixStructureParams,
    pub algorithm: AlgorithmSelectionParams,
    pub memory: MemoryHierarchyParams,
    pub prefetch: PrefetchParams,
    pub accumulator: AccumulatorParams,
}

/// Generates all parameter combinations for exploration
pub struct ParameterSpaceExplorer {
    matrix_space: MatrixStructureSpace,
    algorithm_space: AlgorithmSelectionSpace,
    memory_space: MemoryHierarchySpace,
    rng: ChaCha8Rng,
}

impl ParameterSpaceExplorer {
    pub fn new(seed: u64) -> Self {
        Self {
            matrix_space: MatrixStructureSpace::default(),
            algorithm_space: AlgorithmSelectionSpace::default(),
            memory_space: MemoryHierarchySpace::default(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Generate parameter configurations covering the joint distribution
    pub fn generate_configurations(&mut self) -> Vec<ParameterConfiguration> {
        let mut configs = Vec::new();

        // Clone the spaces to avoid borrowing issues
        let matrix_space = self.matrix_space.clone();
        let algorithm_space = self.algorithm_space.clone();
        let memory_space = self.memory_space.clone();

        // Iterate through matrix structure space
        for &size in &matrix_space.size_points {
            for &density in &matrix_space.density_points {
                for &(nnz_min, nnz_max) in &matrix_space.nnz_ranges {
                    // Skip invalid combinations
                    if !Self::is_valid_matrix_params(size, density, nnz_max) {
                        continue;
                    }

                    // Generate algorithm selection parameters
                    for &dense_factor in &algorithm_space.dense_threshold_factors {
                        for &gpu_factor in &algorithm_space.gpu_threshold_factors {
                            // Generate memory hierarchy parameters
                            for &l2_factor in &memory_space.l2_cache_factors {
                                for &chunk_factor in &memory_space.chunk_size_factors {
                                    configs.push(self.create_configuration(
                                        size,
                                        density,
                                        nnz_min,
                                        nnz_max,
                                        dense_factor,
                                        gpu_factor,
                                        l2_factor,
                                        chunk_factor,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        configs
    }

    /// Check if matrix parameters are valid
    fn is_valid_matrix_params(size: usize, density: f64, max_nnz: usize) -> bool {
        // Ensure density is achievable with given size
        let max_possible_nnz = size;
        let required_nnz = (size as f64 * density) as usize;
        
        required_nnz <= max_possible_nnz && max_nnz <= max_possible_nnz
    }

    /// Create a single parameter configuration
    fn create_configuration(
        &mut self,
        size: usize,
        density: f64,
        nnz_min: usize,
        nnz_max: usize,
        dense_factor: f64,
        gpu_factor: f64,
        l2_factor: f64,
        chunk_factor: f64,
    ) -> ParameterConfiguration {
        let avg_nnz = self.rng.gen_range(nnz_min..=nnz_max);

        ParameterConfiguration {
            matrix: MatrixStructureParams {
                size,
                density,
                avg_nnz_per_row: avg_nnz,
            },
            algorithm: AlgorithmSelectionParams {
                dense_threshold: (GENERIC_DENSE_THRESHOLD as f64 * dense_factor) as usize,
                gpu_threshold: (METAL_GPU_THRESHOLD as f64 * gpu_factor) as usize,
                simd_min_elements: NEON_MIN_ELEMENTS,
            },
            memory: MemoryHierarchyParams {
                cache_line_size: DEFAULT_CACHE_LINE_SIZE,
                l2_cache_size: (DEFAULT_L2_CACHE_SIZE as f64 * l2_factor) as usize,
                chunk_size: (GENERIC_OPTIMAL_CHUNK_SIZE as f64 * chunk_factor) as usize,
                memory_threshold: MODERATE_MEMORY_THRESHOLD,
            },
            prefetch: PrefetchParams {
                distance_threshold: PREFETCH_DISTANCE_THRESHOLD,
                count_multiplier: PREFETCH_COUNT_MULTIPLIER,
                hit_rate_threshold: MEDIUM_HIT_RATE_THRESHOLD,
            },
            accumulator: AccumulatorParams {
                initial_capacity_divisor: INITIAL_CAPACITY_DIVISOR,
                max_capacity: MAX_SORT_ACCUMULATOR_CAPACITY,
                default_size: DEFAULT_SORT_ACCUMULATOR_SIZE,
            },
        }
    }
}

// ============================================================================
// MATRIX GENERATORS
// ============================================================================

/// Generates random sparse matrices based on parameter configuration
pub struct MatrixGenerator {
    rng: ChaCha8Rng,
}

impl MatrixGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Generate NUM_RAND_OBJS random matrices for a given configuration
    pub fn generate_matrices(
        &mut self,
        config: &ParameterConfiguration,
    ) -> Vec<SparseMatrixCSR<f64>> {
        let mut matrices = Vec::with_capacity(NUM_RAND_OBJS);

        for _ in 0..NUM_RAND_OBJS {
            matrices.push(self.generate_single_matrix(config));
        }

        matrices
    }

    /// Generate a single random sparse matrix
    fn generate_single_matrix(&mut self, config: &ParameterConfiguration) -> SparseMatrixCSR<f64> {
        let n = config.matrix.size;
        let density = config.matrix.density;
        let avg_nnz = config.matrix.avg_nnz_per_row;

        // Choose generation strategy based on density
        if density < SPARSE_DENSITY_THRESHOLD {
            self.generate_ultra_sparse_matrix(n, avg_nnz)
        } else if density < SPARSE_DENSITY {
            self.generate_sparse_matrix(n, density)
        } else if density < MEDIUM_DENSITY {
            self.generate_medium_matrix(n, density)
        } else {
            self.generate_dense_matrix(n, density)
        }
    }

    /// Generate ultra-sparse matrix with controlled NNZ per row
    fn generate_ultra_sparse_matrix(&mut self, n: usize, avg_nnz: usize) -> SparseMatrixCSR<f64> {
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        let nnz_dist = Uniform::from(1..=(avg_nnz * 2).min(n));
        let col_dist = Uniform::from(0..n);
        let val_dist = Uniform::from(-10.0..10.0);

        for _ in 0..n {
            let row_nnz = nnz_dist.sample(&mut self.rng);
            let mut row_cols = std::collections::HashSet::new();

            // Generate unique column indices for this row
            while row_cols.len() < row_nnz {
                row_cols.insert(col_dist.sample(&mut self.rng));
            }

            // Sort columns and add to matrix
            let mut sorted_cols: Vec<_> = row_cols.into_iter().collect();
            sorted_cols.sort_unstable();

            for col in sorted_cols {
                col_idx.push(col);
                values.push(val_dist.sample(&mut self.rng));
            }

            row_ptr.push(col_idx.len());
        }

        SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
    }

    /// Generate sparse matrix with given density
    fn generate_sparse_matrix(&mut self, n: usize, density: f64) -> SparseMatrixCSR<f64> {
        let total_nnz = ((n * n) as f64 * density) as usize;
        self.generate_matrix_with_nnz(n, total_nnz)
    }

    /// Generate medium density matrix
    fn generate_medium_matrix(&mut self, n: usize, density: f64) -> SparseMatrixCSR<f64> {
        let total_nnz = ((n * n) as f64 * density) as usize;
        self.generate_matrix_with_nnz(n, total_nnz)
    }

    /// Generate dense matrix (still sparse format but higher density)
    fn generate_dense_matrix(&mut self, n: usize, density: f64) -> SparseMatrixCSR<f64> {
        let total_nnz = ((n * n) as f64 * density) as usize;
        self.generate_matrix_with_nnz(n, total_nnz)
    }

    /// Helper to generate matrix with specific NNZ count
    fn generate_matrix_with_nnz(&mut self, n: usize, total_nnz: usize) -> SparseMatrixCSR<f64> {
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        // Distribute NNZ across rows
        let avg_per_row = total_nnz / n;
        let mut remaining = total_nnz;

        let col_dist = Uniform::from(0..n);
        let val_dist = Uniform::from(-10.0..10.0);

        for i in 0..n {
            let row_nnz = if i == n - 1 {
                remaining
            } else {
                let variation = self.rng.gen_range(0.5..1.5);
                ((avg_per_row as f64 * variation) as usize).min(remaining).min(n)
            };

            let mut row_cols = std::collections::HashSet::new();

            // Generate unique column indices
            while row_cols.len() < row_nnz {
                row_cols.insert(col_dist.sample(&mut self.rng));
            }

            // Sort and add
            let mut sorted_cols: Vec<_> = row_cols.into_iter().collect();
            sorted_cols.sort_unstable();

            for col in sorted_cols {
                col_idx.push(col);
                values.push(val_dist.sample(&mut self.rng));
            }

            remaining = remaining.saturating_sub(row_nnz);
            row_ptr.push(col_idx.len());
        }

        SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
    }
}

// ============================================================================
// SPECIALIZED GENERATORS
// ============================================================================

/// Generates matrices with specific patterns for testing
pub struct PatternMatrixGenerator {
    rng: ChaCha8Rng,
}

impl PatternMatrixGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Generate a banded matrix
    pub fn generate_banded(&mut self, n: usize, bandwidth: usize) -> SparseMatrixCSR<f64> {
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        let val_dist = Uniform::from(-10.0..10.0);

        for i in 0..n {
            let col_start = i.saturating_sub(bandwidth / 2);
            let col_end = (i + bandwidth / 2 + 1).min(n);

            for j in col_start..col_end {
                col_idx.push(j);
                values.push(val_dist.sample(&mut self.rng));
            }

            row_ptr.push(col_idx.len());
        }

        SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
    }

    /// Generate a block diagonal matrix
    pub fn generate_block_diagonal(
        &mut self,
        n: usize,
        block_size: usize,
    ) -> SparseMatrixCSR<f64> {
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        let val_dist = Uniform::from(-10.0..10.0);
        let num_blocks = (n + block_size - 1) / block_size;

        for block_id in 0..num_blocks {
            let block_start = block_id * block_size;
            let block_end = ((block_id + 1) * block_size).min(n);

            for i in block_start..block_end {
                for j in block_start..block_end {
                    if self.rng.gen_bool(0.7) {
                        // 70% density within blocks
                        col_idx.push(j);
                        values.push(val_dist.sample(&mut self.rng));
                    }
                }
                row_ptr.push(col_idx.len());
            }
        }

        // Fill remaining rows if n is not divisible by block_size
        for _ in (num_blocks * block_size)..n {
            row_ptr.push(col_idx.len());
        }

        SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
    }

    /// Generate a power-law distribution matrix (scale-free)
    pub fn generate_power_law(&mut self, n: usize, alpha: f64) -> SparseMatrixCSR<f64> {
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        let val_dist = Uniform::from(-10.0..10.0);

        // Generate degree sequence following power law
        let mut degrees: Vec<usize> = Vec::with_capacity(n);
        for i in 1..=n {
            let degree = ((i as f64).powf(-alpha) * n as f64) as usize;
            degrees.push(degree.max(1).min(n));
        }

        // Shuffle to randomize which nodes get high degree
        use rand::seq::SliceRandom;
        degrees.shuffle(&mut self.rng);

        for degree in degrees {
            let mut row_cols = std::collections::HashSet::new();
            
            while row_cols.len() < degree {
                row_cols.insert(self.rng.gen_range(0..n));
            }

            let mut sorted_cols: Vec<_> = row_cols.into_iter().collect();
            sorted_cols.sort_unstable();

            for col in sorted_cols {
                col_idx.push(col);
                values.push(val_dist.sample(&mut self.rng));
            }

            row_ptr.push(col_idx.len());
        }

        SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
    }
}

// ============================================================================
// TEST SUITE GENERATOR
// ============================================================================

/// Generates complete test suites based on parameter space exploration
pub struct TestSuiteGenerator {
    param_explorer: ParameterSpaceExplorer,
    matrix_gen: MatrixGenerator,
    pattern_gen: PatternMatrixGenerator,
}

impl TestSuiteGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            param_explorer: ParameterSpaceExplorer::new(seed),
            matrix_gen: MatrixGenerator::new(seed + 1),
            pattern_gen: PatternMatrixGenerator::new(seed + 2),
        }
    }

    /// Generate a complete test suite
    pub fn generate_test_suite(&mut self) -> TestSuite {
        let configurations = self.param_explorer.generate_configurations();
        let mut test_cases = Vec::new();

        for config in configurations {
            let matrices = self.matrix_gen.generate_matrices(&config);
            
            test_cases.push(TestCase {
                config: config.clone(),
                matrices,
            });
        }

        // Add pattern-based test cases
        test_cases.extend(self.generate_pattern_tests());

        TestSuite { test_cases }
    }

    /// Generate pattern-based test cases
    fn generate_pattern_tests(&mut self) -> Vec<TestCase> {
        let mut cases = Vec::new();
        let sizes = vec![1000, 5000, 10000];

        for &size in &sizes {
            // Banded matrices
            for bandwidth in [10, 50, 100] {
                if bandwidth < size {
                    cases.push(self.create_pattern_test_case(
                        size,
                        PatternType::Banded(bandwidth),
                    ));
                }
            }

            // Block diagonal matrices
            for block_size in [16, 64, 256] {
                if block_size < size {
                    cases.push(self.create_pattern_test_case(
                        size,
                        PatternType::BlockDiagonal(block_size),
                    ));
                }
            }

            // Power law matrices
            for alpha in [0.5, 1.0, 2.0] {
                cases.push(self.create_pattern_test_case(
                    size,
                    PatternType::PowerLaw(alpha),
                ));
            }
        }

        cases
    }

    fn create_pattern_test_case(&mut self, size: usize, pattern: PatternType) -> TestCase {
        let mut matrices = Vec::with_capacity(NUM_RAND_OBJS);

        for _ in 0..NUM_RAND_OBJS {
            let matrix = match pattern {
                PatternType::Banded(bw) => self.pattern_gen.generate_banded(size, bw),
                PatternType::BlockDiagonal(bs) => {
                    self.pattern_gen.generate_block_diagonal(size, bs)
                }
                PatternType::PowerLaw(alpha) => self.pattern_gen.generate_power_law(size, alpha),
            };
            matrices.push(matrix);
        }

        TestCase {
            config: self.create_default_config(size),
            matrices,
        }
    }

    fn create_default_config(&self, size: usize) -> ParameterConfiguration {
        ParameterConfiguration {
            matrix: MatrixStructureParams {
                size,
                density: SPARSE_DENSITY,
                avg_nnz_per_row: MIN_AVG_NNZ_THRESHOLD,
            },
            algorithm: AlgorithmSelectionParams {
                dense_threshold: GENERIC_DENSE_THRESHOLD,
                gpu_threshold: METAL_GPU_THRESHOLD,
                simd_min_elements: NEON_MIN_ELEMENTS,
            },
            memory: MemoryHierarchyParams {
                cache_line_size: DEFAULT_CACHE_LINE_SIZE,
                l2_cache_size: DEFAULT_L2_CACHE_SIZE,
                chunk_size: GENERIC_OPTIMAL_CHUNK_SIZE,
                memory_threshold: MODERATE_MEMORY_THRESHOLD,
            },
            prefetch: PrefetchParams {
                distance_threshold: PREFETCH_DISTANCE_THRESHOLD,
                count_multiplier: PREFETCH_COUNT_MULTIPLIER,
                hit_rate_threshold: MEDIUM_HIT_RATE_THRESHOLD,
            },
            accumulator: AccumulatorParams {
                initial_capacity_divisor: INITIAL_CAPACITY_DIVISOR,
                max_capacity: MAX_SORT_ACCUMULATOR_CAPACITY,
                default_size: DEFAULT_SORT_ACCUMULATOR_SIZE,
            },
        }
    }
}

#[derive(Debug)]
enum PatternType {
    Banded(usize),
    BlockDiagonal(usize),
    PowerLaw(f64),
}

/// A complete test suite with multiple test cases
pub struct TestSuite {
    pub test_cases: Vec<TestCase>,
}

/// A single test case with configuration and matrices
pub struct TestCase {
    pub config: ParameterConfiguration,
    pub matrices: Vec<SparseMatrixCSR<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_space_generation() {
        let mut explorer = ParameterSpaceExplorer::new(42);
        let configs = explorer.generate_configurations();
        
        assert!(!configs.is_empty());
        println!("Generated {} parameter configurations", configs.len());
    }

    #[test]
    fn test_matrix_generation() {
        let mut gen = MatrixGenerator::new(42);
        let config = ParameterConfiguration {
            matrix: MatrixStructureParams {
                size: 1000,
                density: 0.01,
                avg_nnz_per_row: 10,
            },
            algorithm: AlgorithmSelectionParams {
                dense_threshold: 256,
                gpu_threshold: 10000,
                simd_min_elements: 4,
            },
            memory: MemoryHierarchyParams {
                cache_line_size: 64,
                l2_cache_size: 256_000,
                chunk_size: 256,
                memory_threshold: MODERATE_MEMORY_THRESHOLD,
            },
            prefetch: PrefetchParams {
                distance_threshold: 2,
                count_multiplier: 2,
                hit_rate_threshold: 0.7,
            },
            accumulator: AccumulatorParams {
                initial_capacity_divisor: 10,
                max_capacity: 1024,
                default_size: 256,
            },
        };

        let matrices = gen.generate_matrices(&config);
        assert_eq!(matrices.len(), NUM_RAND_OBJS);

        for matrix in &matrices {
            assert_eq!(matrix.n_rows, 1000);
            assert_eq!(matrix.n_cols, 1000);
            println!("Generated matrix with {} non-zeros", matrix.nnz());
        }
    }

    #[test]
    fn test_pattern_generation() {
        let mut gen = PatternMatrixGenerator::new(42);
        
        // Test banded matrix
        let banded = gen.generate_banded(100, 5);
        assert_eq!(banded.n_rows, 100);
        println!("Banded matrix nnz: {}", banded.nnz());

        // Test block diagonal
        let block_diag = gen.generate_block_diagonal(100, 10);
        assert_eq!(block_diag.n_rows, 100);
        println!("Block diagonal matrix nnz: {}", block_diag.nnz());

        // Test power law
        let power_law = gen.generate_power_law(100, 1.5);
        assert_eq!(power_law.n_rows, 100);
        println!("Power law matrix nnz: {}", power_law.nnz());
    }

    #[test]
    fn test_complete_suite_generation() {
        let mut gen = TestSuiteGenerator::new(42);
        let suite = gen.generate_test_suite();
        
        assert!(!suite.test_cases.is_empty());
        println!("Generated {} test cases", suite.test_cases.len());
        
        // Check that each test case has the right number of matrices
        for case in &suite.test_cases {
            assert_eq!(case.matrices.len(), NUM_RAND_OBJS);
        }
    }
}