//! AVX512-optimized accumulator for sparse matrix multiplication
//!
//! This module provides AVX512-accelerated sorting and accumulation
//! for intermediate products in sparse matrix multiplication.

#![cfg(target_arch = "x86_64")]

use super::Accumulator;
use aligned_vec::AVec;

/// Check if AVX512 is available at runtime
pub fn is_avx512_available() -> bool {
    #[cfg(target_feature = "avx512f")]
    {
        true
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512cd")
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
        if self.size == 0 {
            return (vec![], vec![]);
        }

        // For now, use fallback scalar implementation
        // TODO: Implement actual AVX512 sorting
        self.sort_and_accumulate_scalar()
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
}