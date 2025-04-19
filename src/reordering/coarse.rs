//! Coarse-level reordering implementation (Algorithm 4 in the MAGNUS paper)
//!
//! This module implements the coarse-level reordering approach for improving
//! memory access patterns in sparse matrix multiplication for rows with extremely
//! large intermediate products. It works by first re-organizing matrix A into AˆCSC,
//! then batching rows and using fine-level reordering within each batch.

use num_traits::Num;
use std::ops::AddAssign;

use crate::matrix::SparseMatrixCSR;
use crate::matrix::SparseMatrixCSC;
use crate::matrix::config::MagnusConfig;
use super::{ChunkMetadata, Reordering, exclusive_scan};
// Removed unused import

/// Matrix A in CSC format for coarse-level reordering
pub struct AHatCSC<T> {
    /// Maps to original rows in matrix A
    pub original_row_indices: Vec<usize>,
    
    /// The CSC representation of the matrix
    pub matrix: SparseMatrixCSC<T>,
}

impl<T> AHatCSC<T> 
where 
    T: Copy + Num + AddAssign,
{
    /// Construct AˆCSC from selected rows of A
    ///
    /// # Arguments
    ///
    /// * `a` - Original matrix A in CSR format
    /// * `coarse_rows` - Indices of rows requiring coarse-level reordering
    ///
    /// # Returns
    ///
    /// A new AHatCSC matrix
    pub fn new(a: &SparseMatrixCSR<T>, coarse_rows: &[usize]) -> Self {
        // Create bitmap to track which rows are needed
        let mut row_bitmap = vec![false; a.n_rows];
        for &row in coarse_rows {
            row_bitmap[row] = true;
        }
        
        // Count nonzeros per column
        let mut col_counts = vec![0; a.n_cols];
        
        for &row in coarse_rows {
            let row_start = a.row_ptr[row];
            let row_end = a.row_ptr[row + 1];
            
            for idx in row_start..row_end {
                let col = a.col_idx[idx];
                col_counts[col] += 1;
            }
        }
        
        // Calculate total non-zeros
        let nnz = col_counts.iter().sum();
        
        // Compute column pointers via prefix sum - must create a valid CSC matrix
        // Note: we need one more element than the counts array for prefix sum
        let mut col_ptr = vec![0; a.n_cols + 1];
        
        // Compute exclusive scan 
        let mut running_sum = 0;
        for i in 0..a.n_cols {
            col_ptr[i] = running_sum;
            running_sum += col_counts[i];
        }
        // Last element is total nnz
        col_ptr[a.n_cols] = nnz;
        
        // Allocate arrays for CSC matrix
        let mut row_idx = vec![0; nnz];
        let mut values = vec![T::zero(); nnz];
        
        // Fill CSC matrix (second pass)
        let mut write_pos = col_ptr.clone();
        write_pos.pop(); // Remove the last element for write positions
        
        for row in 0..a.n_rows {
            if !row_bitmap[row] {
                continue;
            }
            
            let row_start = a.row_ptr[row];
            let row_end = a.row_ptr[row + 1];
            
            for idx in row_start..row_end {
                let col = a.col_idx[idx];
                let pos = write_pos[col];
                
                if pos < row_idx.len() {
                    row_idx[pos] = row;
                    values[pos] = a.values[idx];
                    write_pos[col] += 1;
                }
            }
        }
        
        // Create the CSC matrix
        let matrix = SparseMatrixCSC::new(
            a.n_rows,
            a.n_cols,
            col_ptr,
            row_idx,
            values,
        );
        
        Self {
            original_row_indices: coarse_rows.to_vec(),
            matrix,
        }
    }
}

/// Holds temporary data structures for coarse-level reordering
pub struct CoarseLevelReordering {
    /// Metadata about the chunk size and count
    metadata: ChunkMetadata,
    
    /// Maximum batch size for processing
    batch_size: usize,
}

impl CoarseLevelReordering {
    /// Create a new coarse-level reordering instance
    ///
    /// # Arguments
    ///
    /// * `n_cols` - Number of columns in the output matrix
    /// * `config` - MAGNUS configuration parameters
    ///
    /// # Returns
    ///
    /// A new CoarseLevelReordering instance
    pub fn new(n_cols: usize, config: &MagnusConfig) -> Self {
        // Determine an appropriate batch size based on available memory
        // This is a simplification - real implementation would consider more factors
        let batch_size = determine_batch_size(config);
        
        Self {
            metadata: ChunkMetadata::new(n_cols, config),
            batch_size,
        }
    }

    /// Process a batch of rows using coarse-level reordering
    ///
    /// # Arguments
    ///
    /// * `a_hat_csc` - Matrix A in CSC format
    /// * `b` - Matrix B in CSR format
    /// * `start_idx` - Start index in the batch
    /// * `end_idx` - End index in the batch
    /// * `config` - Configuration parameters
    ///
    /// # Returns
    ///
    /// A vector of (row, col_indices, values) tuples for each processed row
    fn process_batch<T>(&self,
                   a_hat_csc: &AHatCSC<T>,
                   b: &SparseMatrixCSR<T>,
                   start_idx: usize,
                   end_idx: usize,
                   _config: &MagnusConfig) -> Vec<(usize, Vec<usize>, Vec<T>)>
    where
        T: Copy + Num + AddAssign,
    {
        let batch_size = end_idx - start_idx;
        if batch_size == 0 {
            return Vec::new();
        }
        
        // For each chunk of columns in B
        let mut results = vec![(0, Vec::new(), Vec::new()); batch_size];
        
        // Initialize row results with the correct row indices
        for i in 0..batch_size {
            let batch_idx = start_idx + i;
            if batch_idx < a_hat_csc.original_row_indices.len() {
                let row = a_hat_csc.original_row_indices[batch_idx];
                results[i].0 = row;
            }
        }
        
        // Process each chunk of columns
        for chunk_idx in 0..self.metadata.n_chunks {
            let chunk_start = chunk_idx * self.metadata.chunk_length;
            let chunk_end = std::cmp::min(chunk_start + self.metadata.chunk_length, b.n_cols);
            
            self.process_column_chunk(
                a_hat_csc,
                b,
                start_idx,
                end_idx,
                chunk_start,
                chunk_end,
                &mut results,
            );
        }
        
        results
    }
    
    /// Process a chunk of columns for a batch of rows
    ///
    /// # Arguments
    ///
    /// * `a_hat_csc` - Matrix A in CSC format
    /// * `b` - Matrix B in CSR format
    /// * `start_idx` - Start index in the batch
    /// * `end_idx` - End index in the batch
    /// * `chunk_start` - Start column of the chunk
    /// * `chunk_end` - End column of the chunk
    /// * `results` - Output results for each row
    fn process_column_chunk<T>(&self,
                          a_hat_csc: &AHatCSC<T>,
                          b: &SparseMatrixCSR<T>,
                          start_idx: usize,
                          end_idx: usize,
                          chunk_start: usize,
                          chunk_end: usize,
                          results: &mut [(usize, Vec<usize>, Vec<T>)])
    where
        T: Copy + Num + AddAssign,
    {
        // Create temporary accumulators for this chunk
        let batch_size = end_idx - start_idx;
        let mut chunk_accumulators = vec![Vec::<(usize, T)>::new(); batch_size];
        
        // Process each column in the chunk
        for col_idx in chunk_start..chunk_end {
            // Get the elements in column col_idx of A
            let col_start = a_hat_csc.matrix.col_ptr[col_idx];
            let col_end = a_hat_csc.matrix.col_ptr[col_idx + 1];
            
            if col_start == col_end {
                continue; // Empty column
            }
            
            // Get the elements in row col_idx of B
            let b_row_start = b.row_ptr[col_idx];
            let b_row_end = b.row_ptr[col_idx + 1];
            
            if b_row_start == b_row_end {
                continue; // Empty row in B
            }
            
            // For each element in A[:, col_idx]
            for a_idx in col_start..col_end {
                let a_row = a_hat_csc.matrix.row_idx[a_idx];
                let a_val = a_hat_csc.matrix.values[a_idx];
                
                // Find the batch index for this row
                let batch_idx_opt = a_hat_csc.original_row_indices[start_idx..end_idx]
                    .iter()
                    .position(|&r| r == a_row);
                
                if let Some(batch_idx) = batch_idx_opt {
                    // For each element in B[col_idx, :]
                    for b_idx in b_row_start..b_row_end {
                        let b_col = b.col_idx[b_idx];
                        let b_val = b.values[b_idx];
                        
                        let product = a_val * b_val;
                        chunk_accumulators[batch_idx].push((b_col, product));
                    }
                }
            }
        }
        
        // Process accumulators for this chunk
        for i in 0..batch_size {
            if chunk_accumulators[i].is_empty() {
                continue;
            }
            
            // Sort by column
            chunk_accumulators[i].sort_by_key(|&(col, _)| col);
            
            // Merge duplicates
            let mut j = 0;
            while j < chunk_accumulators[i].len() - 1 {
                let current_col = chunk_accumulators[i][j].0;
                let next_col = chunk_accumulators[i][j + 1].0;
                
                if current_col == next_col {
                    // Fix borrow issue by getting the value first
                    let next_val = chunk_accumulators[i][j + 1].1;
                    chunk_accumulators[i][j].1 += next_val;
                    chunk_accumulators[i].remove(j + 1);
                } else {
                    j += 1;
                }
            }
            
            // Extract columns and values
            let chunk_cols: Vec<usize> = chunk_accumulators[i].iter().map(|&(col, _)| col).collect();
            let chunk_vals: Vec<T> = chunk_accumulators[i].iter().map(|&(_, val)| val).collect();
            
            // Merge into results (ensuring columns remain sorted)
            if results[i].1.is_empty() {
                results[i].1 = chunk_cols;
                results[i].2 = chunk_vals;
            } else {
                // Merge the sorted arrays
                let (merged_cols, merged_vals) = merge_sorted_arrays(
                    &results[i].1, &results[i].2,
                    &chunk_cols, &chunk_vals,
                );
                
                results[i].1 = merged_cols;
                results[i].2 = merged_vals;
            }
        }
    }
}

impl<T> Reordering<T> for CoarseLevelReordering
where
    T: Copy + Num + AddAssign,
{
    fn multiply_row(
        &self,
        a_row: usize,
        a: &SparseMatrixCSR<T>,
        b: &SparseMatrixCSR<T>,
        config: &MagnusConfig,
    ) -> (Vec<usize>, Vec<T>) {
        // For coarse-level reordering, we need to process this row alongside others
        // Here we just create AˆCSC for this single row and use our batch processing
        
        // Create AˆCSC just for this row
        let a_hat_csc = AHatCSC::new(a, &[a_row]);
        
        // Process this single row as a batch
        let results = self.process_batch(&a_hat_csc, b, 0, 1, config);
        
        if results.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            // Return the results for this row
            (results[0].1.clone(), results[0].2.clone())
        }
    }
}

/// Process a set of rows using coarse-level reordering (Algorithm 4)
///
/// # Arguments
///
/// * `a` - Matrix A in CSR format
/// * `b` - Matrix B in CSR format
/// * `rows` - Row indices requiring coarse-level reordering
/// * `config` - Configuration parameters
///
/// # Returns
///
/// A vector of (row, col_indices, values) tuples for each processed row
pub fn process_coarse_level_rows<T>(
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    rows: &[usize],
    config: &MagnusConfig,
) -> Vec<(usize, Vec<usize>, Vec<T>)>
where
    T: Copy + Num + AddAssign,
{
    if rows.is_empty() {
        return Vec::new();
    }
    
    // Create AˆCSC from the specified rows
    let a_hat_csc = AHatCSC::new(a, rows);
    
    // Create the reordering instance
    let reordering = CoarseLevelReordering::new(b.n_cols, config);
    
    // Process in batches
    let mut results = Vec::new();
    let n_rows = rows.len();
    
    for start_idx in (0..n_rows).step_by(reordering.batch_size) {
        let end_idx = std::cmp::min(start_idx + reordering.batch_size, n_rows);
        
        let batch_results = reordering.process_batch(
            &a_hat_csc,
            b,
            start_idx,
            end_idx,
            config,
        );
        
        results.extend(batch_results);
    }
    
    results
}

/// Process a single row using coarse-level reordering (Algorithm 4)
///
/// This is a convenience wrapper for processing a single row.
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
/// A tuple of (column indices, values) for the result row
pub fn multiply_row_coarse_level<T>(
    a_row: usize,
    a: &SparseMatrixCSR<T>,
    b: &SparseMatrixCSR<T>,
    config: &MagnusConfig,
) -> (Vec<usize>, Vec<T>)
where
    T: Copy + Num + AddAssign,
{
    let reordering = CoarseLevelReordering::new(b.n_cols, config);
    reordering.multiply_row(a_row, a, b, config)
}

/// Determine appropriate batch size for coarse-level reordering
///
/// # Arguments
///
/// * `config` - Configuration parameters
///
/// # Returns
///
/// An appropriate batch size
fn determine_batch_size(config: &MagnusConfig) -> usize {
    // In a real implementation, this would be more sophisticated
    // considering available memory, cache sizes, etc.
    
    // For now, use a simple heuristic based on L2 cache size
    let cache_size = config.system_params.l2_cache_size;
    
    // Estimate memory required per row
    let estimated_memory_per_row = 64; // Just a placeholder value
    
    // Calculate how many rows we can process together
    let cache_based_size = cache_size / estimated_memory_per_row;
    
    // Ensure at least one row and at most some reasonable limit
    std::cmp::max(1, std::cmp::min(cache_based_size, 64))
}

/// Merge two sorted arrays while accumulating duplicate entries
///
/// # Arguments
///
/// * `cols1` - First array of column indices
/// * `vals1` - First array of values
/// * `cols2` - Second array of column indices
/// * `vals2` - Second array of values
///
/// # Returns
///
/// Merged (columns, values) with accumulated duplicates
fn merge_sorted_arrays<T>(
    cols1: &[usize], 
    vals1: &[T],
    cols2: &[usize], 
    vals2: &[T],
) -> (Vec<usize>, Vec<T>)
where
    T: Copy + Num + AddAssign,
{
    let mut result_cols = Vec::with_capacity(cols1.len() + cols2.len());
    let mut result_vals = Vec::with_capacity(vals1.len() + vals2.len());
    
    let mut i = 0;
    let mut j = 0;
    
    // Merge the two sorted arrays
    while i < cols1.len() && j < cols2.len() {
        match cols1[i].cmp(&cols2[j]) {
            std::cmp::Ordering::Less => {
                result_cols.push(cols1[i]);
                result_vals.push(vals1[i]);
                i += 1;
            },
            std::cmp::Ordering::Greater => {
                result_cols.push(cols2[j]);
                result_vals.push(vals2[j]);
                j += 1;
            },
            std::cmp::Ordering::Equal => {
                // Same column - accumulate values
                let mut sum = vals1[i];
                sum += vals2[j];
                
                result_cols.push(cols1[i]);
                result_vals.push(sum);
                
                i += 1;
                j += 1;
            }
        }
    }
    
    // Add remaining elements from first array
    while i < cols1.len() {
        result_cols.push(cols1[i]);
        result_vals.push(vals1[i]);
        i += 1;
    }
    
    // Add remaining elements from second array
    while j < cols2.len() {
        result_cols.push(cols2[j]);
        result_vals.push(vals2[j]);
        j += 1;
    }
    
    (result_cols, result_vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::config::{MagnusConfig, SystemParameters, Architecture};
    
    #[test]
    fn test_merge_sorted_arrays() {
        let cols1 = vec![1, 3, 5, 7];
        let vals1 = vec![1.0f64, 2.0, 3.0, 4.0];
        
        let cols2 = vec![2, 3, 6, 7, 8];
        let vals2 = vec![5.0f64, 6.0, 7.0, 8.0, 9.0];
        
        let (merged_cols, merged_vals) = merge_sorted_arrays(&cols1, &vals1, &cols2, &vals2);
        
        // Expected: cols [1, 2, 3, 5, 6, 7, 8]
        // Expected: vals [1.0, 5.0, 8.0, 3.0, 7.0, 12.0, 9.0]
        assert_eq!(merged_cols, vec![1, 2, 3, 5, 6, 7, 8]);
        
        // Check the values with a small epsilon for floating point comparisons
        let epsilon = 1e-10;
        assert!((merged_vals[0] - 1.0).abs() < epsilon);
        assert!((merged_vals[1] - 5.0).abs() < epsilon);
        assert!((merged_vals[2] - 8.0).abs() < epsilon); // 2.0 + 6.0
        assert!((merged_vals[3] - 3.0).abs() < epsilon);
        assert!((merged_vals[4] - 7.0).abs() < epsilon);
        assert!((merged_vals[5] - 12.0).abs() < epsilon); // 4.0 + 8.0
        assert!((merged_vals[6] - 9.0).abs() < epsilon);
    }
    
    #[test]
    fn test_a_hat_csc_construction() {
        // Create a test matrix A
        // A = [1 2 0; 0 3 4; 5 0 6]
        let a = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        
        // Select rows 0 and 2 for coarse-level reordering
        let coarse_rows = vec![0, 2];
        
        // Create AˆCSC
        let a_hat_csc = AHatCSC::new(&a, &coarse_rows);
        
        // Check original_row_indices
        assert_eq!(a_hat_csc.original_row_indices, vec![0, 2]);
        
        // Check CSC matrix structure
        // Col 0 should have entries for rows 0 and 2 with values 1.0 and 5.0
        // Col 1 should have an entry for row 0 with value 2.0
        // Col 2 should have an entry for row 2 with value 6.0
        
        // Check column pointers
        assert_eq!(a_hat_csc.matrix.col_ptr, vec![0, 2, 3, 4]);
        
        // Check row indices and values for column 0
        let col0_indices: Vec<usize> = a_hat_csc.matrix.row_idx[0..2].to_vec();
        let col0_values: Vec<f64> = a_hat_csc.matrix.values[0..2].to_vec();
        
        // Sort by row index for deterministic comparison
        let mut col0_pairs: Vec<(usize, f64)> = col0_indices.iter()
            .zip(col0_values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();
        col0_pairs.sort_by_key(|&(idx, _)| idx);
        
        assert_eq!(col0_pairs.len(), 2);
        assert_eq!(col0_pairs[0].0, 0); // Row 0
        assert_eq!(col0_pairs[1].0, 2); // Row 2
        
        let epsilon = 1e-10;
        assert!((col0_pairs[0].1 - 1.0).abs() < epsilon); // Value for row 0
        assert!((col0_pairs[1].1 - 5.0).abs() < epsilon); // Value for row 2
        
        // Check remaining columns
        assert_eq!(a_hat_csc.matrix.row_idx[2], 0); // Row 0, Col 1
        assert!((a_hat_csc.matrix.values[2] - 2.0).abs() < epsilon);
        
        assert_eq!(a_hat_csc.matrix.row_idx[3], 2); // Row 2, Col 2
        assert!((a_hat_csc.matrix.values[3] - 6.0).abs() < epsilon);
    }
    
    #[test]
    fn test_coarse_level_single_row() {
        // Create test matrices:
        // A = [1 2 0; 0 3 4; 5 0 6]
        // B = [7 0 0; 0 8 0; 0 0 9]
        // Expected result:
        // C = [7 16 0; 0 24 36; 35 0 54]
        
        let a = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        
        let b = SparseMatrixCSR::new(
            3, 3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![7.0f64, 8.0, 9.0],
        );
        
        let config = MagnusConfig::default();
        
        // Test the coarse-level reordering for each row
        let (cols_row0, vals_row0) = multiply_row_coarse_level(0, &a, &b, &config);
        let (cols_row1, vals_row1) = multiply_row_coarse_level(1, &a, &b, &config);
        let (cols_row2, vals_row2) = multiply_row_coarse_level(2, &a, &b, &config);
        
        // Verify row 0 has expected results
        assert_eq!(cols_row0, vec![0, 1]);
        let diff1: f64 = (vals_row0[0] - 7.0f64).abs();
        let diff2: f64 = (vals_row0[1] - 16.0f64).abs();
        assert!(diff1 < 1.0e-10);
        assert!(diff2 < 1.0e-10);
        
        // Verify row 1 has expected results
        assert_eq!(cols_row1, vec![1, 2]);
        let diff3: f64 = (vals_row1[0] - 24.0f64).abs();
        let diff4: f64 = (vals_row1[1] - 36.0f64).abs();
        assert!(diff3 < 1.0e-10);
        assert!(diff4 < 1.0e-10);
        
        // Verify row 2 has expected results
        assert_eq!(cols_row2, vec![0, 2]);
        let diff5: f64 = (vals_row2[0] - 35.0f64).abs();
        let diff6: f64 = (vals_row2[1] - 54.0f64).abs();
        assert!(diff5 < 1.0e-10);
        assert!(diff6 < 1.0e-10);
    }
    
    #[test]
    fn test_coarse_level_batch_processing() {
        // Create test matrices:
        // A = [1 2 0; 0 3 4; 5 0 6]
        // B = [7 0 0; 0 8 0; 0 0 9]
        // Expected result:
        // C = [7 16 0; 0 24 36; 35 0 54]
        
        let a = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        
        let b = SparseMatrixCSR::new(
            3, 3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![7.0f64, 8.0, 9.0],
        );
        
        let config = MagnusConfig::default();
        
        // Process all rows using coarse-level reordering
        let rows = vec![0, 1, 2];
        let results = process_coarse_level_rows(&a, &b, &rows, &config);
        
        // Check that we got results for all rows
        assert_eq!(results.len(), 3);
        
        // Find result for each row
        let mut row0_result = None;
        let mut row1_result = None;
        let mut row2_result = None;
        
        for (row, cols, vals) in &results {
            match row {
                0 => row0_result = Some((cols, vals)),
                1 => row1_result = Some((cols, vals)),
                2 => row2_result = Some((cols, vals)),
                _ => panic!("Unexpected row index"),
            }
        }
        
        // Verify row 0 has expected results
        let (cols_row0, vals_row0) = row0_result.unwrap();
        assert_eq!(*cols_row0, vec![0, 1]);
        let diff1: f64 = (vals_row0[0] - 7.0f64).abs();
        let diff2: f64 = (vals_row0[1] - 16.0f64).abs();
        assert!(diff1 < 1.0e-10);
        assert!(diff2 < 1.0e-10);
        
        // Verify row 1 has expected results
        let (cols_row1, vals_row1) = row1_result.unwrap();
        assert_eq!(*cols_row1, vec![1, 2]);
        let diff3: f64 = (vals_row1[0] - 24.0f64).abs();
        let diff4: f64 = (vals_row1[1] - 36.0f64).abs();
        assert!(diff3 < 1.0e-10);
        assert!(diff4 < 1.0e-10);
        
        // Verify row 2 has expected results
        let (cols_row2, vals_row2) = row2_result.unwrap();
        assert_eq!(*cols_row2, vec![0, 2]);
        let diff5: f64 = (vals_row2[0] - 35.0f64).abs();
        let diff6: f64 = (vals_row2[1] - 54.0f64).abs();
        assert!(diff5 < 1.0e-10);
        assert!(diff6 < 1.0e-10);
    }
    
    #[test]
    fn test_empty_row() {
        // Test with an empty row
        let a = SparseMatrixCSR::new(
            2, 2,
            vec![0, 0, 1], // Row 0 is empty
            vec![0],
            vec![1.0f64],
        );
        
        let b = SparseMatrixCSR::new(
            2, 2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![2.0f64, 3.0],
        );
        
        let config = MagnusConfig::default();
        
        let (cols, vals) = multiply_row_coarse_level(0, &a, &b, &config);
        
        // Result should be empty
        assert_eq!(cols.len(), 0);
        assert_eq!(vals.len(), 0);
    }
    
    #[test]
    fn test_larger_matrix() {
        // Create a larger test case
        let n = 10;
        
        // Create a diagonal matrix A
        let mut a_row_ptr = vec![0];
        let mut a_col_idx = Vec::new();
        let mut a_values = Vec::new();
        
        for i in 0..n {
            a_col_idx.push(i);
            a_values.push(1.0f64);
            a_row_ptr.push(i + 1);
        }
        
        let a = SparseMatrixCSR::new(
            n, n,
            a_row_ptr,
            a_col_idx,
            a_values,
        );
        
        // Create a diagonal matrix B
        let mut b_row_ptr = vec![0];
        let mut b_col_idx = Vec::new();
        let mut b_values = Vec::new();
        
        for i in 0..n {
            b_col_idx.push(i);
            b_values.push(2.0f64);
            b_row_ptr.push(i + 1);
        }
        
        let b = SparseMatrixCSR::new(
            n, n,
            b_row_ptr,
            b_col_idx,
            b_values,
        );
        
        let config = MagnusConfig::default();
        
        // Process all rows using coarse-level reordering
        let mut rows = Vec::new();
        for i in 0..n {
            rows.push(i);
        }
        
        let results = process_coarse_level_rows(&a, &b, &rows, &config);
        
        // Check that we got results for all rows
        assert_eq!(results.len(), n);
        
        // Each row i should have a single nonzero entry at column i with value 2.0
        for (row, cols, vals) in &results {
            assert_eq!(cols.len(), 1);
            assert_eq!(cols[0], *row);
            let diff: f64 = (vals[0] - 2.0f64).abs();
            assert!(diff < 1.0e-10);
        }
    }
}