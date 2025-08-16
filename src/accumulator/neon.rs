//! Optimized ARM NEON implementation for sorting and accumulation
//! 
//! This module provides NEON-accelerated bitonic sorting networks
//! for various input sizes, optimized for Apple Silicon processors.

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use std::arch::aarch64::*;
use super::SimdAccelerator;

/// Compare and swap specific lanes (macro for const indices)
macro_rules! compare_swap_lanes {
    ($idx:expr, $val:expr, $lane_a:literal, $lane_b:literal) => {
        unsafe {
            let idx_a = vgetq_lane_u32($idx, $lane_a);
            let idx_b = vgetq_lane_u32($idx, $lane_b);
            
            if idx_a > idx_b {
                // Swap indices
                $idx = vsetq_lane_u32(idx_b, $idx, $lane_a);
                $idx = vsetq_lane_u32(idx_a, $idx, $lane_b);
                
                // Swap values
                let val_a = vgetq_lane_f32($val, $lane_a);
                let val_b = vgetq_lane_f32($val, $lane_b);
                $val = vsetq_lane_f32(val_b, $val, $lane_a);
                $val = vsetq_lane_f32(val_a, $val, $lane_b);
            }
        }
    };
}

/// NEON accumulator with size-specific optimizations
pub struct NeonAccumulator;

impl NeonAccumulator {
    pub fn new() -> Self {
        NeonAccumulator
    }
}

impl SimdAccelerator<f32> for NeonAccumulator {
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let len = col_indices.len();
        
        // Dynamic strategy selection based on input size
        match len {
            0 => (Vec::new(), Vec::new()),
            1..=3 => scalar_sort_accumulate(col_indices, values),
            4 => unsafe { neon_sort_4_exact(col_indices, values) },
            5..=8 => unsafe { neon_sort_8_padded(col_indices, values) },
            9..=16 => unsafe { neon_sort_16_padded(col_indices, values) },
            17..=32 => unsafe { neon_sort_32_padded(col_indices, values) },
            33..=64 => unsafe { neon_hybrid_sort(col_indices, values) },
            _ => {
                // For larger sizes, use chunked approach or fallback
                super::FallbackAccumulator.sort_and_accumulate(col_indices, values)
            }
        }
    }
}

/// Scalar implementation for very small inputs
fn scalar_sort_accumulate(col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
    if col_indices.is_empty() {
        return (Vec::new(), Vec::new());
    }
    
    let mut pairs: Vec<_> = col_indices.iter()
        .zip(values.iter())
        .map(|(&idx, &val)| (idx, val))
        .collect();
    
    pairs.sort_by_key(|&(idx, _)| idx);
    
    // Accumulate duplicates
    let mut result_indices = Vec::new();
    let mut result_values = Vec::new();
    
    let mut current_idx = pairs[0].0;
    let mut current_sum = pairs[0].1;
    
    for &(idx, val) in &pairs[1..] {
        if idx == current_idx {
            current_sum += val;
        } else {
            result_indices.push(current_idx);
            result_values.push(current_sum);
            current_idx = idx;
            current_sum = val;
        }
    }
    
    result_indices.push(current_idx);
    result_values.push(current_sum);
    
    (result_indices, result_values)
}

/// NEON bitonic sort for exactly 4 elements
#[target_feature(enable = "neon")]
unsafe fn neon_sort_4_exact(col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
    debug_assert_eq!(col_indices.len(), 4);
    debug_assert_eq!(values.len(), 4);
    
    // Convert indices to u32 for NEON
    let mut idx = [
        col_indices[0] as u32,
        col_indices[1] as u32,
        col_indices[2] as u32,
        col_indices[3] as u32,
    ];
    let mut val = [values[0], values[1], values[2], values[3]];
    
    // Load into NEON registers
    let mut idx_vec = vld1q_u32(idx.as_ptr());
    let mut val_vec = vld1q_f32(val.as_ptr());
    
    // Bitonic sort network for 4 elements (6 comparisons)
    // Stage 1: Compare 0-2, 1-3
    compare_swap_lanes!(idx_vec, val_vec, 0, 2);
    compare_swap_lanes!(idx_vec, val_vec, 1, 3);
    
    // Stage 2: Compare 0-1, 2-3
    compare_swap_lanes!(idx_vec, val_vec, 0, 1);
    compare_swap_lanes!(idx_vec, val_vec, 2, 3);
    
    // Stage 3: Compare 1-2
    compare_swap_lanes!(idx_vec, val_vec, 1, 2);
    
    // Store back
    vst1q_u32(idx.as_mut_ptr(), idx_vec);
    vst1q_f32(val.as_mut_ptr(), val_vec);
    
    // Accumulate duplicates
    accumulate_sorted(&idx, &val)
}

/// NEON bitonic sort for up to 8 elements (with padding)
#[target_feature(enable = "neon")]
unsafe fn neon_sort_8_padded(col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let len = col_indices.len();
    debug_assert!(len <= 8);
    
    // Pad with maximum values
    let mut idx = [u32::MAX; 8];
    let mut val = [f32::INFINITY; 8];
    
    for i in 0..len {
        idx[i] = col_indices[i] as u32;
        val[i] = values[i];
    }
    
    // Perform 8-element bitonic sort
    neon_bitonic_sort_8(&mut idx, &mut val);
    
    // Remove padding and accumulate
    accumulate_sorted(&idx[..len], &val[..len])
}

/// Core 8-element bitonic sort network
#[inline(always)]
unsafe fn neon_bitonic_sort_8(idx: &mut [u32; 8], val: &mut [f32; 8]) {
    // Load as two vectors
    let mut idx_lo = vld1q_u32(&idx[0]);
    let mut idx_hi = vld1q_u32(&idx[4]);
    let mut val_lo = vld1q_f32(&val[0]);
    let mut val_hi = vld1q_f32(&val[4]);
    
    // Stage 1: Compare across halves
    compare_swap_vectors(&mut idx_lo, &mut idx_hi, &mut val_lo, &mut val_hi);
    
    // Stage 2: Sort each half independently
    bitonic_sort_4_inplace(&mut idx_lo, &mut val_lo);
    bitonic_sort_4_inplace(&mut idx_hi, &mut val_hi);
    
    // Stage 3: Final merge
    // This requires more complex merging logic
    // For now, store and use scalar merge
    vst1q_u32(&mut idx[0], idx_lo);
    vst1q_u32(&mut idx[4], idx_hi);
    vst1q_f32(&mut val[0], val_lo);
    vst1q_f32(&mut val[4], val_hi);
    
    // Final merge step (can be optimized further)
    merge_sorted_halves(idx, val);
}

/// NEON bitonic sort for up to 16 elements
#[target_feature(enable = "neon")]
unsafe fn neon_sort_16_padded(col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let len = col_indices.len();
    debug_assert!(len <= 16);
    
    // Pad with maximum values
    let mut idx = [u32::MAX; 16];
    let mut val = [f32::INFINITY; 16];
    
    for i in 0..len {
        idx[i] = col_indices[i] as u32;
        val[i] = values[i];
    }
    
    // Sort 4 groups of 4
    for i in 0..4 {
        let offset = i * 4;
        let mut idx_chunk = [idx[offset], idx[offset+1], idx[offset+2], idx[offset+3]];
        let mut val_chunk = [val[offset], val[offset+1], val[offset+2], val[offset+3]];
        
        let idx_vec = vld1q_u32(idx_chunk.as_ptr());
        let val_vec = vld1q_f32(val_chunk.as_ptr());
        
        let (sorted_idx, sorted_val) = bitonic_sort_4(idx_vec, val_vec);
        
        vst1q_u32(idx_chunk.as_mut_ptr(), sorted_idx);
        vst1q_f32(val_chunk.as_mut_ptr(), sorted_val);
        
        idx[offset..offset+4].copy_from_slice(&idx_chunk);
        val[offset..offset+4].copy_from_slice(&val_chunk);
    }
    
    // Merge sorted chunks
    merge_chunks_16(&mut idx, &mut val);
    
    // Remove padding and accumulate
    accumulate_sorted(&idx[..len], &val[..len])
}

/// NEON sort for up to 32 elements
#[target_feature(enable = "neon")]
unsafe fn neon_sort_32_padded(col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let len = col_indices.len();
    debug_assert!(len <= 32);
    
    // Pad with maximum values
    let mut idx = [u32::MAX; 32];
    let mut val = [f32::INFINITY; 32];
    
    for i in 0..len {
        idx[i] = col_indices[i] as u32;
        val[i] = values[i];
    }
    
    // Sort 8 groups of 4 using NEON
    for i in 0..8 {
        let offset = i * 4;
        let mut idx_chunk = [idx[offset], idx[offset+1], idx[offset+2], idx[offset+3]];
        let mut val_chunk = [val[offset], val[offset+1], val[offset+2], val[offset+3]];
        
        let idx_vec = vld1q_u32(idx_chunk.as_ptr());
        let val_vec = vld1q_f32(val_chunk.as_ptr());
        
        let (sorted_idx, sorted_val) = bitonic_sort_4(idx_vec, val_vec);
        
        vst1q_u32(idx_chunk.as_mut_ptr(), sorted_idx);
        vst1q_f32(val_chunk.as_mut_ptr(), sorted_val);
        
        idx[offset..offset+4].copy_from_slice(&idx_chunk);
        val[offset..offset+4].copy_from_slice(&val_chunk);
    }
    
    // Hierarchical merge: 8->4->2->1
    merge_chunks_32(&mut idx, &mut val);
    
    // Remove padding and accumulate
    accumulate_sorted(&idx[..len], &val[..len])
}

/// Hybrid approach for medium sizes (33-64 elements)
#[target_feature(enable = "neon")]
unsafe fn neon_hybrid_sort(col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let len = col_indices.len();
    debug_assert!(len > 32 && len <= 64);
    
    // Process in chunks of 16 using NEON
    let chunk_size = 16;
    let num_chunks = (len + chunk_size - 1) / chunk_size;
    
    let mut all_indices = Vec::with_capacity(len);
    let mut all_values = Vec::with_capacity(len);
    
    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(len);
        let chunk_indices = &col_indices[start..end];
        let chunk_values = &values[start..end];
        
        let (sorted_idx, sorted_val) = neon_sort_16_padded(chunk_indices, chunk_values);
        all_indices.extend(sorted_idx);
        all_values.extend(sorted_val);
    }
    
    // Final merge if we had multiple chunks
    if num_chunks > 1 {
        let mut pairs: Vec<_> = all_indices.iter()
            .zip(all_values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();
        pairs.sort_by_key(|&(idx, _)| idx);
        
        // Accumulate
        let mut result_indices = Vec::new();
        let mut result_values = Vec::new();
        
        let mut current_idx = pairs[0].0;
        let mut current_sum = pairs[0].1;
        
        for &(idx, val) in &pairs[1..] {
            if idx == current_idx {
                current_sum += val;
            } else {
                result_indices.push(current_idx);
                result_values.push(current_sum);
                current_idx = idx;
                current_sum = val;
            }
        }
        
        result_indices.push(current_idx);
        result_values.push(current_sum);
        
        (result_indices, result_values)
    } else {
        (all_indices, all_values)
    }
}

// Helper functions

/// Compare and swap entire vectors
#[inline(always)]
unsafe fn compare_swap_vectors(
    idx_a: &mut uint32x4_t,
    idx_b: &mut uint32x4_t,
    val_a: &mut float32x4_t,
    val_b: &mut float32x4_t,
) {
    let mask = vcgtq_u32(*idx_a, *idx_b);
    
    // Swap indices based on mask
    let idx_min = vbslq_u32(mask, *idx_b, *idx_a);
    let idx_max = vbslq_u32(mask, *idx_a, *idx_b);
    
    // Swap values based on same mask
    let val_min = vbslq_f32(mask, *val_b, *val_a);
    let val_max = vbslq_f32(mask, *val_a, *val_b);
    
    *idx_a = idx_min;
    *idx_b = idx_max;
    *val_a = val_min;
    *val_b = val_max;
}

/// Bitonic sort 4 elements in a vector
#[inline(always)]
unsafe fn bitonic_sort_4(idx: uint32x4_t, val: float32x4_t) -> (uint32x4_t, float32x4_t) {
    let mut idx = idx;
    let mut val = val;
    
    // 6 comparisons for 4-element sort
    compare_swap_lanes!(idx, val, 0, 2);
    compare_swap_lanes!(idx, val, 1, 3);
    compare_swap_lanes!(idx, val, 0, 1);
    compare_swap_lanes!(idx, val, 2, 3);
    compare_swap_lanes!(idx, val, 1, 2);
    
    (idx, val)
}

/// In-place bitonic sort for 4 elements
#[inline(always)]
unsafe fn bitonic_sort_4_inplace(idx: &mut uint32x4_t, val: &mut float32x4_t) {
    compare_swap_lanes!(*idx, *val, 0, 2);
    compare_swap_lanes!(*idx, *val, 1, 3);
    compare_swap_lanes!(*idx, *val, 0, 1);
    compare_swap_lanes!(*idx, *val, 2, 3);
    compare_swap_lanes!(*idx, *val, 1, 2);
}

/// Merge two sorted halves of an 8-element array using NEON
#[target_feature(enable = "neon")]
unsafe fn merge_sorted_halves(idx: &mut [u32; 8], val: &mut [f32; 8]) {
    // Load both halves
    let idx_lo = vld1q_u32(&idx[0]);
    let idx_hi = vld1q_u32(&idx[4]);
    let _val_lo = vld1q_f32(&val[0]);
    let _val_hi = vld1q_f32(&val[4]);
    
    // Use NEON min/max to create sorted result
    // This is a simplified merge network
    let mut temp_idx = [0u32; 8];
    let mut temp_val = [0.0f32; 8];
    
    // Extract first elements from each half for comparison
    let lo_first = vgetq_lane_u32(idx_lo, 0);
    let hi_first = vgetq_lane_u32(idx_hi, 0);
    
    if lo_first <= hi_first {
        // Low half has smaller first element
        // Merge with NEON comparisons
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        
        while i < 4 && j < 4 {
            let lo_val = idx[i];
            let hi_val = idx[4 + j];
            
            if lo_val <= hi_val {
                temp_idx[k] = lo_val;
                temp_val[k] = val[i];
                i += 1;
            } else {
                temp_idx[k] = hi_val;
                temp_val[k] = val[4 + j];
                j += 1;
            }
            k += 1;
        }
        
        // Copy remaining elements
        while i < 4 {
            temp_idx[k] = idx[i];
            temp_val[k] = val[i];
            i += 1;
            k += 1;
        }
        
        while j < 4 {
            temp_idx[k] = idx[4 + j];
            temp_val[k] = val[4 + j];
            j += 1;
            k += 1;
        }
    } else {
        // High half has smaller first element
        // Similar merge logic
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        
        while i < 4 && j < 4 {
            let lo_val = idx[i];
            let hi_val = idx[4 + j];
            
            if lo_val <= hi_val {
                temp_idx[k] = lo_val;
                temp_val[k] = val[i];
                i += 1;
            } else {
                temp_idx[k] = hi_val;
                temp_val[k] = val[4 + j];
                j += 1;
            }
            k += 1;
        }
        
        while i < 4 {
            temp_idx[k] = idx[i];
            temp_val[k] = val[i];
            i += 1;
            k += 1;
        }
        
        while j < 4 {
            temp_idx[k] = idx[4 + j];
            temp_val[k] = val[4 + j];
            j += 1;
            k += 1;
        }
    }
    
    *idx = temp_idx;
    *val = temp_val;
}

/// Merge 4 sorted chunks of 4 elements each
fn merge_chunks_16(idx: &mut [u32; 16], val: &mut [f32; 16]) {
    // Merge pairs of chunks
    let mut temp_idx = [0u32; 16];
    let mut temp_val = [0.0f32; 16];
    
    // Merge chunks 0&1, 2&3
    merge_two_chunks(&idx[0..8], &val[0..8], &mut temp_idx[0..8], &mut temp_val[0..8]);
    merge_two_chunks(&idx[8..16], &val[8..16], &mut temp_idx[8..16], &mut temp_val[8..16]);
    
    // Final merge
    merge_two_chunks(&temp_idx[0..16], &temp_val[0..16], idx, val);
}

/// Merge two sorted chunks
fn merge_two_chunks(
    idx_in: &[u32],
    val_in: &[f32],
    idx_out: &mut [u32],
    val_out: &mut [f32],
) {
    let mid = idx_in.len() / 2;
    let mut i = 0;
    let mut j = mid;
    let mut k = 0;
    
    while i < mid && j < idx_in.len() {
        if idx_in[i] <= idx_in[j] {
            idx_out[k] = idx_in[i];
            val_out[k] = val_in[i];
            i += 1;
        } else {
            idx_out[k] = idx_in[j];
            val_out[k] = val_in[j];
            j += 1;
        }
        k += 1;
    }
    
    while i < mid {
        idx_out[k] = idx_in[i];
        val_out[k] = val_in[i];
        i += 1;
        k += 1;
    }
    
    while j < idx_in.len() {
        idx_out[k] = idx_in[j];
        val_out[k] = val_in[j];
        j += 1;
        k += 1;
    }
}

/// Merge 8 sorted chunks of 4 elements each (for 32-element sort)
fn merge_chunks_32(idx: &mut [u32; 32], val: &mut [f32; 32]) {
    // Stage 1: Merge pairs (8 chunks -> 4 chunks)
    let mut temp_idx = [0u32; 32];
    let mut temp_val = [0.0f32; 32];
    
    for i in 0..4 {
        let src_offset = i * 8;
        let dst_offset = i * 8;
        merge_two_chunks(
            &idx[src_offset..src_offset + 8],
            &val[src_offset..src_offset + 8],
            &mut temp_idx[dst_offset..dst_offset + 8],
            &mut temp_val[dst_offset..dst_offset + 8],
        );
    }
    
    *idx = temp_idx;
    *val = temp_val;
    
    // Stage 2: Merge pairs (4 chunks -> 2 chunks)
    for i in 0..2 {
        let src_offset = i * 16;
        let dst_offset = i * 16;
        merge_two_chunks(
            &idx[src_offset..src_offset + 16],
            &val[src_offset..src_offset + 16],
            &mut temp_idx[dst_offset..dst_offset + 16],
            &mut temp_val[dst_offset..dst_offset + 16],
        );
    }
    
    *idx = temp_idx;
    *val = temp_val;
    
    // Stage 3: Final merge (2 chunks -> 1 chunk)
    merge_two_chunks(&idx[0..32], &val[0..32], &mut temp_idx, &mut temp_val);
    
    *idx = temp_idx;
    *val = temp_val;
}

/// Accumulate sorted data
fn accumulate_sorted(indices: &[u32], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
    if indices.is_empty() {
        return (Vec::new(), Vec::new());
    }
    
    let mut result_idx = Vec::new();
    let mut result_val = Vec::new();
    
    let mut current_idx = indices[0];
    let mut current_sum = values[0];
    
    for i in 1..indices.len() {
        if indices[i] == current_idx {
            current_sum += values[i];
        } else {
            if current_idx != u32::MAX { // Skip padding
                result_idx.push(current_idx as usize);
                result_val.push(current_sum);
            }
            current_idx = indices[i];
            current_sum = values[i];
        }
    }
    
    if current_idx != u32::MAX {
        result_idx.push(current_idx as usize);
        result_val.push(current_sum);
    }
    
    (result_idx, result_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neon_sort_4() {
        let indices = vec![3, 1, 4, 2];
        let values = vec![3.0, 1.0, 4.0, 2.0];
        
        let acc = NeonAccumulator::new();
        let (sorted_idx, sorted_val) = acc.sort_and_accumulate(&indices, &values);
        
        assert_eq!(sorted_idx, vec![1, 2, 3, 4]);
        assert_eq!(sorted_val, vec![1.0, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_neon_accumulation() {
        let indices = vec![2, 1, 2, 1];
        let values = vec![2.0, 1.0, 3.0, 4.0];
        
        let acc = NeonAccumulator::new();
        let (sorted_idx, sorted_val) = acc.sort_and_accumulate(&indices, &values);
        
        assert_eq!(sorted_idx, vec![1, 2]);
        assert_eq!(sorted_val, vec![5.0, 5.0]); // 1+4=5, 2+3=5
    }
}