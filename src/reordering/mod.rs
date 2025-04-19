//! Reordering strategies for improved memory locality in sparse matrix operations
//!
//! This module implements the reordering strategies described in the MAGNUS paper:
//! - Fine-level reordering (Algorithm 3)
//! - Coarse-level reordering (Algorithm 4)
//!
//! These strategies improve memory access patterns and cache utilization,
//! especially for rows with large intermediate products.

pub mod fine;
pub mod coarse;

use num_traits::Num;
use std::ops::AddAssign;

use crate::matrix::SparseMatrixCSR;
use crate::matrix::config::MagnusConfig;

/// Metadata for chunking operations in reordering algorithms
pub struct ChunkMetadata {
    /// Size of each chunk (in elements)
    pub chunk_length: usize,
    
    /// Number of chunks
    pub n_chunks: usize,
    
    /// Number of bits to shift right to get chunk index (for power-of-2 chunk sizes)
    pub shift_bits: usize,
}

impl ChunkMetadata {
    /// Create a new chunk metadata object with optimal settings
    ///
    /// # Arguments
    ///
    /// * `total_elements` - Total number of elements to divide into chunks
    /// * `config` - MAGNUS configuration with cache parameters
    ///
    /// # Returns
    ///
    /// A ChunkMetadata object with appropriate chunk settings for the given matrix
    pub fn new(total_elements: usize, config: &MagnusConfig) -> Self {
        // Calculate chunk size based on L2 cache size
        // We aim for chunks that fit comfortably in L2 cache
        let _cache_line_size = config.system_params.cache_line_size; // May be used for future optimizations
        let l2_cache_size = config.system_params.l2_cache_size;
        
        // Aim for chunks that are power of 2 and fit well in cache
        // Default to a reasonable value (64 elements) if cache info isn't available
        let chunk_length = if l2_cache_size > 0 {
            // Use largest power of 2 that allows multiple chunks to fit in L2 cache
            // This is a simplification - actual calculation would consider more factors
            let target_size = l2_cache_size / 8;  // Use 1/8th of L2 cache
            let elements_per_chunk = target_size / std::mem::size_of::<usize>();
            
            // Find nearest power of 2
            let mut pow2 = 64;  // Start with a reasonable minimum
            while pow2 * 2 <= elements_per_chunk {
                pow2 *= 2;
            }
            pow2
        } else {
            // Default if cache info not available
            64
        };
        
        // Calculate number of chunks needed
        let n_chunks = (total_elements + chunk_length - 1) / chunk_length;
        
        // Calculate shift bits for power-of-2 chunk size
        // This is a fast way to compute the chunk index: col >> shift_bits
        let shift_bits = chunk_length.trailing_zeros() as usize;
        
        Self {
            chunk_length,
            n_chunks,
            shift_bits,
        }
    }
    
    /// Calculate the chunk index for a given column
    ///
    /// # Arguments
    ///
    /// * `col` - Column index
    ///
    /// # Returns
    ///
    /// The chunk index for the column
    #[inline]
    pub fn get_chunk_index(&self, col: usize) -> usize {
        // Use bitshift for power-of-2 chunk sizes (faster than division)
        col >> self.shift_bits
    }
}

/// Generic trait for reordering strategies
///
/// This trait defines the interface for different reordering strategies,
/// allowing them to be used interchangeably where appropriate.
pub trait Reordering<T>
where
    T: Copy + Num + AddAssign,
{
    /// Reorder a row multiplication to improve locality
    ///
    /// # Arguments
    ///
    /// * `a_row` - Row index in matrix A
    /// * `a` - Matrix A in CSR format
    /// * `b` - Matrix B in CSR format
    /// * `config` - Configuration parameters
    ///
    /// # Returns
    ///
    /// A tuple of `(col_indices, values)` containing the non-zero entries of the result row.
    fn multiply_row(
        &self,
        a_row: usize,
        a: &SparseMatrixCSR<T>,
        b: &SparseMatrixCSR<T>,
        config: &MagnusConfig,
    ) -> (Vec<usize>, Vec<T>);
}

// Helper function to perform exclusive scan (prefix sum)
pub(crate) fn exclusive_scan(input: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(input.len() + 1);
    result.push(0);  // Start with 0
    
    let mut running_sum = 0;
    for &val in input {
        result.push(running_sum);
        running_sum += val;
    }
    
    result
}

// Re-export key functions for convenient access
pub use fine::multiply_row_fine_level;
pub use coarse::{multiply_row_coarse_level, process_coarse_level_rows};