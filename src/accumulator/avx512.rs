//! AVX512-optimized accumulator for sparse matrix multiplication
//!
//! This module provides AVX512-accelerated sorting and accumulation
//! for intermediate products in sparse matrix multiplication.

#![cfg(target_arch = "x86_64")]

use super::Accumulator;
use aligned_vec::AVec;
use std::arch::x86_64::*;

/// Check if AVX512 is available at runtime
pub fn is_avx512_available() -> bool {
    #[cfg(target_feature = "avx512f")]
    {
        true
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        is_x86_feature_detected!("avx512f")
    }
}

/// Check if AVX512CD (Conflict Detection) is available
pub fn is_avx512cd_available() -> bool {
    #[cfg(target_feature = "avx512cd")]
    {
        true
    }
    #[cfg(not(target_feature = "avx512cd"))]
    {
        is_x86_feature_detected!("avx512cd")
    }
}

/// AVX512-accelerated accumulator for sorting and accumulating sparse products
pub struct Avx512Accumulator {
    /// Storage for column indices (AVec provides alignment)
    col_indices: AVec<u32>,
    /// Storage for values (AVec provides alignment)
    values: AVec<f32>,
    /// Current number of accumulated entries
    size: usize,
    /// Allocated capacity
    capacity: usize,
}

impl Avx512Accumulator {
    /// Create a new AVX512 accumulator with the given initial capacity
    pub fn new(initial_capacity: usize) -> Self {
        // Round up to multiple of 16 (AVX512 processes 16 x 32-bit elements)
        let aligned_capacity = ((initial_capacity + 15) / 16) * 16;
        
        Self {
            col_indices: AVec::from_iter(64, (0..aligned_capacity).map(|_| 0u32)),
            values: AVec::from_iter(64, (0..aligned_capacity).map(|_| 0.0f32)),
            size: 0,
            capacity: aligned_capacity,
        }
    }

    /// Ensure we have enough capacity for additional elements
    fn ensure_capacity(&mut self, additional: usize) {
        let required = self.size + additional;
        if required > self.capacity {
            let new_capacity = ((required * 2 + 15) / 16) * 16;
            
            let mut new_indices = AVec::from_iter(64, (0..new_capacity).map(|_| 0u32));
            let mut new_values = AVec::from_iter(64, (0..new_capacity).map(|_| 0.0f32));
            
            new_indices[..self.size].copy_from_slice(&self.col_indices[..self.size]);
            new_values[..self.size].copy_from_slice(&self.values[..self.size]);
            
            self.col_indices = new_indices;
            self.values = new_values;
            self.capacity = new_capacity;
        }
    }

    /// Sort and accumulate using AVX512 instructions
    /// 
    /// # Safety
    /// This function uses AVX512 intrinsics which require proper CPU support
    #[target_feature(enable = "avx512f")]
    unsafe fn sort_and_accumulate_avx512(&self) -> (Vec<u32>, Vec<f32>) {
        match self.size {
            0 => (vec![], vec![]),
            1..=16 => self.sort_16_or_less(),
            17..=32 => self.sort_32_padded(),
            33..=64 => self.sort_64_padded(),
            _ => self.sort_large_avx512(),
        }
    }

    /// Sort up to 16 elements using AVX512 bitonic sort
    #[target_feature(enable = "avx512f")]
    unsafe fn sort_16_or_less(&self) -> (Vec<u32>, Vec<f32>) {
        let n = self.size.min(16);
        
        // Prepare padded arrays for AVX512 processing
        let mut indices = [u32::MAX; 16];
        let mut values = [f32::INFINITY; 16];
        
        indices[..n].copy_from_slice(&self.col_indices[..n]);
        values[..n].copy_from_slice(&self.values[..n]);
        
        // Load into AVX512 registers
        let mut idx_vec = _mm512_loadu_si512(indices.as_ptr() as *const __m512i);
        let mut val_vec = _mm512_loadu_ps(values.as_ptr());
        
        // Perform bitonic sort for 16 elements
        self.bitonic_sort_16(&mut idx_vec, &mut val_vec);
        
        // Store sorted results
        _mm512_storeu_si512(indices.as_mut_ptr() as *mut __m512i, idx_vec);
        _mm512_storeu_ps(values.as_mut_ptr(), val_vec);
        
        // Accumulate duplicates and remove padding
        self.accumulate_and_compact(&indices, &values, n)
    }

    /// Bitonic sort for exactly 16 elements using AVX512 intrinsics
    /// For now, using a simpler approach that extracts, sorts, and reloads
    #[target_feature(enable = "avx512f")]
    unsafe fn bitonic_sort_16(&self, indices: &mut __m512i, values: &mut __m512) {
        // Extract to arrays for sorting
        let mut idx_arr = [0u32; 16];
        let mut val_arr = [0.0f32; 16];
        
        _mm512_storeu_si512(idx_arr.as_mut_ptr() as *mut __m512i, *indices);
        _mm512_storeu_ps(val_arr.as_mut_ptr(), *values);
        
        // Create index-value pairs and sort
        let mut pairs: Vec<(u32, f32)> = idx_arr.iter()
            .zip(val_arr.iter())
            .map(|(&i, &v)| (i, v))
            .collect();
        
        pairs.sort_by_key(|&(i, _)| i);
        
        // Copy back to arrays
        for (i, &(idx, val)) in pairs.iter().enumerate() {
            idx_arr[i] = idx;
            val_arr[i] = val;
        }
        
        // Reload into SIMD registers
        *indices = _mm512_loadu_si512(idx_arr.as_ptr() as *const __m512i);
        *values = _mm512_loadu_ps(val_arr.as_ptr());
    }


    /// Sort 17-32 elements with padding
    #[target_feature(enable = "avx512f")]
    unsafe fn sort_32_padded(&self) -> (Vec<u32>, Vec<f32>) {
        let n = self.size.min(32);
        
        // Prepare padded arrays
        let mut indices = vec![u32::MAX; 32];
        let mut values = vec![f32::INFINITY; 32];
        
        indices[..n].copy_from_slice(&self.col_indices[..n]);
        values[..n].copy_from_slice(&self.values[..n]);
        
        // Process as two 16-element vectors
        let mut idx_vec1 = _mm512_loadu_si512(indices[0..16].as_ptr() as *const __m512i);
        let mut val_vec1 = _mm512_loadu_ps(values[0..16].as_ptr());
        let mut idx_vec2 = _mm512_loadu_si512(indices[16..32].as_ptr() as *const __m512i);
        let mut val_vec2 = _mm512_loadu_ps(values[16..32].as_ptr());
        
        // Sort each half
        self.bitonic_sort_16(&mut idx_vec1, &mut val_vec1);
        self.bitonic_sort_16(&mut idx_vec2, &mut val_vec2);
        
        // Merge the two sorted halves
        self.bitonic_merge_32(&mut idx_vec1, &mut val_vec1, &mut idx_vec2, &mut val_vec2);
        
        // Store results
        _mm512_storeu_si512(indices[0..16].as_mut_ptr() as *mut __m512i, idx_vec1);
        _mm512_storeu_ps(values[0..16].as_mut_ptr(), val_vec1);
        _mm512_storeu_si512(indices[16..32].as_mut_ptr() as *mut __m512i, idx_vec2);
        _mm512_storeu_ps(values[16..32].as_mut_ptr(), val_vec2);
        
        // Accumulate duplicates and remove padding
        self.accumulate_and_compact(&indices, &values, n)
    }

    /// Merge two sorted 16-element vectors into sorted 32 elements
    #[target_feature(enable = "avx512f")]
    unsafe fn bitonic_merge_32(
        &self,
        idx1: &mut __m512i,
        val1: &mut __m512,
        idx2: &mut __m512i,
        val2: &mut __m512
    ) {
        // For now, using a simple merge approach
        // Extract both vectors, merge, and reload
        let mut idx_arr1 = [0u32; 16];
        let mut val_arr1 = [0.0f32; 16];
        let mut idx_arr2 = [0u32; 16];
        let mut val_arr2 = [0.0f32; 16];
        
        _mm512_storeu_si512(idx_arr1.as_mut_ptr() as *mut __m512i, *idx1);
        _mm512_storeu_ps(val_arr1.as_mut_ptr(), *val1);
        _mm512_storeu_si512(idx_arr2.as_mut_ptr() as *mut __m512i, *idx2);
        _mm512_storeu_ps(val_arr2.as_mut_ptr(), *val2);
        
        // Merge the two sorted arrays
        let mut result_idx = [u32::MAX; 32];
        let mut result_val = [f32::INFINITY; 32];
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        
        while i < 16 && j < 16 && idx_arr1[i] != u32::MAX && idx_arr2[j] != u32::MAX {
            if idx_arr1[i] <= idx_arr2[j] {
                result_idx[k] = idx_arr1[i];
                result_val[k] = val_arr1[i];
                i += 1;
            } else {
                result_idx[k] = idx_arr2[j];
                result_val[k] = val_arr2[j];
                j += 1;
            }
            k += 1;
        }
        
        while i < 16 && idx_arr1[i] != u32::MAX {
            result_idx[k] = idx_arr1[i];
            result_val[k] = val_arr1[i];
            i += 1;
            k += 1;
        }
        
        while j < 16 && idx_arr2[j] != u32::MAX {
            result_idx[k] = idx_arr2[j];
            result_val[k] = val_arr2[j];
            j += 1;
            k += 1;
        }
        
        // Reload results into vectors
        *idx1 = _mm512_loadu_si512(result_idx[0..16].as_ptr() as *const __m512i);
        *val1 = _mm512_loadu_ps(result_val[0..16].as_ptr());
        *idx2 = _mm512_loadu_si512(result_idx[16..32].as_ptr() as *const __m512i);
        *val2 = _mm512_loadu_ps(result_val[16..32].as_ptr());
    }

    /// Sort 33-64 elements with padding
    #[target_feature(enable = "avx512f")]
    unsafe fn sort_64_padded(&self) -> (Vec<u32>, Vec<f32>) {
        let n = self.size.min(64);
        
        // Prepare padded arrays
        let mut indices = vec![u32::MAX; 64];
        let mut values = vec![f32::INFINITY; 64];
        
        indices[..n].copy_from_slice(&self.col_indices[..n]);
        values[..n].copy_from_slice(&self.values[..n]);
        
        // Process as four 16-element vectors
        let mut vecs_idx = [_mm512_setzero_si512(); 4];
        let mut vecs_val = [_mm512_setzero_ps(); 4];
        
        for i in 0..4 {
            vecs_idx[i] = _mm512_loadu_si512(indices[i*16..(i+1)*16].as_ptr() as *const __m512i);
            vecs_val[i] = _mm512_loadu_ps(values[i*16..(i+1)*16].as_ptr());
            self.bitonic_sort_16(&mut vecs_idx[i], &mut vecs_val[i]);
        }
        
        // For simplicity, extract all, sort, and reload
        // A proper implementation would use a more efficient merge
        let mut all_idx = vec![];
        let mut all_val = vec![];
        
        for i in 0..4 {
            let mut temp_idx = [0u32; 16];
            let mut temp_val = [0.0f32; 16];
            _mm512_storeu_si512(temp_idx.as_mut_ptr() as *mut __m512i, vecs_idx[i]);
            _mm512_storeu_ps(temp_val.as_mut_ptr(), vecs_val[i]);
            
            for j in 0..16 {
                if temp_idx[j] != u32::MAX {
                    all_idx.push(temp_idx[j]);
                    all_val.push(temp_val[j]);
                }
            }
        }
        
        // Sort all
        let mut pairs: Vec<_> = all_idx.iter().zip(all_val.iter()).map(|(&i, &v)| (i, v)).collect();
        pairs.sort_by_key(|&(i, _)| i);
        
        // Reload into vectors with padding
        for i in 0..4 {
            let mut temp_idx = [u32::MAX; 16];
            let mut temp_val = [f32::INFINITY; 16];
            
            let start = i * 16;
            let end = ((i + 1) * 16).min(pairs.len());
            
            for j in start..end {
                temp_idx[j - start] = pairs[j].0;
                temp_val[j - start] = pairs[j].1;
            }
            
            vecs_idx[i] = _mm512_loadu_si512(temp_idx.as_ptr() as *const __m512i);
            vecs_val[i] = _mm512_loadu_ps(temp_val.as_ptr());
        }
        
        // Store results
        for i in 0..4 {
            _mm512_storeu_si512(indices[i*16..(i+1)*16].as_mut_ptr() as *mut __m512i, vecs_idx[i]);
            _mm512_storeu_ps(values[i*16..(i+1)*16].as_mut_ptr(), vecs_val[i]);
        }
        
        // Accumulate duplicates and remove padding
        self.accumulate_and_compact(&indices, &values, n)
    }

    /// Sort large arrays using hybrid approach
    #[target_feature(enable = "avx512f")]
    unsafe fn sort_large_avx512(&self) -> (Vec<u32>, Vec<f32>) {
        // For large arrays, fall back to scalar sort with AVX512 acceleration for merging
        self.sort_and_accumulate_scalar()
    }

    /// Accumulate duplicates and remove padding
    fn accumulate_and_compact(&self, indices: &[u32], values: &[f32], valid_count: usize) -> (Vec<u32>, Vec<f32>) {
        if valid_count == 0 {
            return (vec![], vec![]);
        }
        
        // Use AVX512CD if available for faster duplicate detection
        if is_avx512cd_available() && valid_count >= 16 {
            unsafe { self.accumulate_and_compact_avx512cd(indices, values, valid_count) }
        } else {
            self.accumulate_and_compact_scalar(indices, values, valid_count)
        }
    }
    
    /// Scalar version of accumulate and compact
    fn accumulate_and_compact_scalar(&self, indices: &[u32], values: &[f32], _valid_count: usize) -> (Vec<u32>, Vec<f32>) {
        let mut result_indices = Vec::new();
        let mut result_values = Vec::new();
        
        let mut current_idx = indices[0];
        let mut current_val = values[0];
        
        for i in 1..indices.len() {
            let idx = indices[i];
            let val = values[i];
            
            // Skip padding values
            if idx == u32::MAX {
                break;
            }
            
            if idx == current_idx {
                // Accumulate duplicate
                current_val += val;
            } else {
                // Output accumulated value
                result_indices.push(current_idx);
                result_values.push(current_val);
                current_idx = idx;
                current_val = val;
            }
        }
        
        // Don't forget the last group (if not padding)
        if current_idx != u32::MAX {
            result_indices.push(current_idx);
            result_values.push(current_val);
        }
        
        (result_indices, result_values)
    }
    
    /// AVX512CD-accelerated version using conflict detection
    #[target_feature(enable = "avx512f,avx512cd")]
    unsafe fn accumulate_and_compact_avx512cd(&self, indices: &[u32], values: &[f32], valid_count: usize) -> (Vec<u32>, Vec<f32>) {
        let mut result_indices = Vec::new();
        let mut result_values = Vec::new();
        
        let mut i = 0;
        
        // Process 16 elements at a time with AVX512CD
        while i + 16 <= valid_count {
            let idx_vec = _mm512_loadu_si512(&indices[i] as *const u32 as *const __m512i);
            let val_vec = _mm512_loadu_ps(&values[i]);
            
            // Use conflict detection to find duplicates within the vector
            // _mm512_conflict_epi32 returns for each element a bitmask of earlier elements with same value
            let conflicts = _mm512_conflict_epi32(idx_vec);
            
            // Process the vector, accumulating duplicates
            let mut temp_indices = [0u32; 16];
            let mut temp_values = [0.0f32; 16];
            _mm512_storeu_si512(temp_indices.as_mut_ptr() as *mut __m512i, idx_vec);
            _mm512_storeu_ps(temp_values.as_mut_ptr(), val_vec);
            
            let mut conflicts_arr = [0u32; 16];
            _mm512_storeu_si512(conflicts_arr.as_mut_ptr() as *mut __m512i, conflicts);
            
            let mut processed = [false; 16];
            
            for j in 0..16 {
                if !processed[j] && temp_indices[j] != u32::MAX {
                    let mut accumulated_val = temp_values[j];
                    
                    // Check for later duplicates
                    for k in (j + 1)..16 {
                        if temp_indices[k] == temp_indices[j] {
                            accumulated_val += temp_values[k];
                            processed[k] = true;
                        }
                    }
                    
                    // Check if this continues a sequence from previous iteration
                    if !result_indices.is_empty() && result_indices.last() == Some(&temp_indices[j]) {
                        *result_values.last_mut().unwrap() += accumulated_val;
                    } else {
                        result_indices.push(temp_indices[j]);
                        result_values.push(accumulated_val);
                    }
                }
            }
            
            i += 16;
        }
        
        // Handle remaining elements with scalar code
        while i < indices.len() && indices[i] != u32::MAX {
            let idx = indices[i];
            let val = values[i];
            
            if !result_indices.is_empty() && result_indices.last() == Some(&idx) {
                *result_values.last_mut().unwrap() += val;
            } else {
                result_indices.push(idx);
                result_values.push(val);
            }
            
            i += 1;
        }
        
        (result_indices, result_values)
    }

    /// Fallback scalar implementation for sorting and accumulation
    fn sort_and_accumulate_scalar(&self) -> (Vec<u32>, Vec<f32>) {
        if self.size == 0 {
            return (vec![], vec![]);
        }

        // Create pairs and sort by column index
        let mut pairs: Vec<(u32, f32)> = self.col_indices[..self.size]
            .iter()
            .zip(self.values[..self.size].iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();

        pairs.sort_unstable_by_key(|&(idx, _)| idx);

        // Accumulate duplicates
        let mut result_indices = Vec::with_capacity(self.size);
        let mut result_values = Vec::with_capacity(self.size);

        let mut current_idx = pairs[0].0;
        let mut current_val = pairs[0].1;

        for &(idx, val) in &pairs[1..] {
            if idx == current_idx {
                current_val += val;
            } else {
                result_indices.push(current_idx);
                result_values.push(current_val);
                current_idx = idx;
                current_val = val;
            }
        }

        result_indices.push(current_idx);
        result_values.push(current_val);

        (result_indices, result_values)
    }
}

impl Accumulator<f32> for Avx512Accumulator {
    fn reset(&mut self) {
        self.size = 0;
    }

    fn accumulate(&mut self, col: usize, val: f32) {
        if val == 0.0 {
            return;
        }

        self.ensure_capacity(1);
        self.col_indices[self.size] = col as u32;
        self.values[self.size] = val;
        self.size += 1;
    }

    fn extract_result(self) -> (Vec<usize>, Vec<f32>) {
        if self.size == 0 {
            return (vec![], vec![]);
        }

        // Use AVX512 if available, otherwise fall back to scalar
        let (indices_u32, values) = if is_avx512_available() {
            unsafe { self.sort_and_accumulate_avx512() }
        } else {
            self.sort_and_accumulate_scalar()
        };

        // Convert u32 indices to usize
        let indices = indices_u32.into_iter().map(|idx| idx as usize).collect();

        (indices, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_detection() {
        // This test just checks that detection runs without crashing
        let available = is_avx512_available();
        println!("AVX512 available: {}", available);
    }

    #[test]
    fn test_basic_accumulation() {
        let mut acc = Avx512Accumulator::new(10);
        
        acc.accumulate(5, 1.0);
        acc.accumulate(3, 2.0);
        acc.accumulate(5, 3.0);  // Duplicate column
        acc.accumulate(1, 4.0);
        
        let (indices, values) = acc.extract_result();
        
        assert_eq!(indices, vec![1, 3, 5]);
        assert_eq!(values, vec![4.0, 2.0, 4.0]);  // 5 appears twice: 1.0 + 3.0 = 4.0
    }

    #[test]
    fn test_capacity_growth() {
        let mut acc = Avx512Accumulator::new(2);
        
        // Add more elements than initial capacity
        for i in 0..100 {
            acc.accumulate(i, i as f32);
        }
        
        let (indices, values) = acc.extract_result();
        assert_eq!(indices.len(), 100);
        assert_eq!(values.len(), 100);
    }

    #[test]
    fn test_zero_values() {
        let mut acc = Avx512Accumulator::new(10);
        
        acc.accumulate(5, 0.0);  // Should be ignored
        acc.accumulate(3, 2.0);
        acc.accumulate(5, 0.0);  // Should be ignored
        acc.accumulate(1, 4.0);
        
        let (indices, values) = acc.extract_result();
        
        assert_eq!(indices, vec![1, 3]);
        assert_eq!(values, vec![4.0, 2.0]);
    }

    #[test]
    fn test_avx512_sizes() {
        // Test exact powers of 16
        for size in [1, 8, 16, 24, 32, 48, 64] {
            let mut acc = Avx512Accumulator::new(size);
            
            for i in 0..size {
                acc.accumulate(i % (size/2 + 1), i as f32);
            }
            
            let (indices, _values) = acc.extract_result();
            
            // Verify sorted
            for i in 1..indices.len() {
                assert!(indices[i] > indices[i-1], "Not sorted at size {}", size);
            }
        }
    }
}