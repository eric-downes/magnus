//! AVX512 bitonic sort implementation
//!
//! This module implements a vectorized bitonic sorting network for 16 elements
//! using AVX512 intrinsics.

use std::arch::x86_64::*;

/// Perform a single compare-exchange operation between two AVX512 vectors
/// Returns (min_vector, max_vector) where elements are sorted pairwise
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn compare_exchange_vectors(
    indices_a: __m512i,
    values_a: __m512,
    indices_b: __m512i,
    values_b: __m512,
) -> (__m512i, __m512, __m512i, __m512) {
    // Compare indices: mask is 1 where a > b
    let cmp_mask = _mm512_cmpgt_epu32_mask(indices_a, indices_b);

    // Select minimum indices and corresponding values
    let min_indices = _mm512_mask_blend_epi32(cmp_mask, indices_a, indices_b);
    let min_values = _mm512_mask_blend_ps(cmp_mask, values_a, values_b);

    // Select maximum indices and corresponding values
    let max_indices = _mm512_mask_blend_epi32(cmp_mask, indices_b, indices_a);
    let max_values = _mm512_mask_blend_ps(cmp_mask, values_b, values_a);

    (min_indices, min_values, max_indices, max_values)
}

/// Bitonic sort for exactly 16 elements using explicit network
/// This implements the full sorting network without dynamic direction calculation
#[target_feature(enable = "avx512f")]
pub unsafe fn bitonic_sort_16_explicit(indices: &mut __m512i, values: &mut __m512) {
    // We'll implement this as a series of compare-exchange operations
    // For 16 elements, we need log2(16) = 4 stages
    // Total comparisons: 1 + 2 + 3 + 4 = 10 passes

    // Convert to arrays for easier manipulation during development
    let mut idx_arr = [0u32; 16];
    let mut val_arr = [0.0f32; 16];

    _mm512_storeu_si512(idx_arr.as_mut_ptr() as *mut __m512i, *indices);
    _mm512_storeu_ps(val_arr.as_mut_ptr(), *values);

    // Implement the classic bitonic sort network for 16 elements
    // Stage 1: Build 2-element bitonic sequences
    for i in 0..8 {
        let idx1 = i * 2;
        let idx2 = i * 2 + 1;
        if idx_arr[idx1] > idx_arr[idx2] {
            idx_arr.swap(idx1, idx2);
            val_arr.swap(idx1, idx2);
        }
    }

    // Stage 2: Build 4-element bitonic sequences
    // First merge pairs in alternating order
    for i in 0..4 {
        let base = i * 4;
        // Sort [0,3] and [1,2] within each group of 4
        if i % 2 == 0 {
            // Ascending order for even groups
            if idx_arr[base] > idx_arr[base + 3] {
                idx_arr.swap(base, base + 3);
                val_arr.swap(base, base + 3);
            }
            if idx_arr[base + 1] > idx_arr[base + 2] {
                idx_arr.swap(base + 1, base + 2);
                val_arr.swap(base + 1, base + 2);
            }
        } else {
            // Descending order for odd groups
            if idx_arr[base] < idx_arr[base + 3] {
                idx_arr.swap(base, base + 3);
                val_arr.swap(base, base + 3);
            }
            if idx_arr[base + 1] < idx_arr[base + 2] {
                idx_arr.swap(base + 1, base + 2);
                val_arr.swap(base + 1, base + 2);
            }
        }
    }

    // Complete stage 2 with adjacent comparisons
    for i in 0..4 {
        let base = i * 4;
        // Sort adjacent pairs
        if idx_arr[base] > idx_arr[base + 1] {
            idx_arr.swap(base, base + 1);
            val_arr.swap(base, base + 1);
        }
        if idx_arr[base + 2] > idx_arr[base + 3] {
            idx_arr.swap(base + 2, base + 3);
            val_arr.swap(base + 2, base + 3);
        }
        // Middle pair
        if idx_arr[base + 1] > idx_arr[base + 2] {
            idx_arr.swap(base + 1, base + 2);
            val_arr.swap(base + 1, base + 2);
        }
    }

    // Stage 3: Build 8-element bitonic sequences
    for i in 0..2 {
        let base = i * 8;

        // Compare elements 4 apart
        for j in 0..4 {
            let idx1 = base + j;
            let idx2 = base + j + 4;

            if i == 0 {
                // First half: ascending
                if idx_arr[idx1] > idx_arr[idx2] {
                    idx_arr.swap(idx1, idx2);
                    val_arr.swap(idx1, idx2);
                }
            } else {
                // Second half: descending
                if idx_arr[idx1] < idx_arr[idx2] {
                    idx_arr.swap(idx1, idx2);
                    val_arr.swap(idx1, idx2);
                }
            }
        }

        // Compare elements 2 apart
        for j in 0..8 {
            if (j & 2) == 0 {
                let idx1 = base + j;
                let idx2 = base + j + 2;
                if idx_arr[idx1] > idx_arr[idx2] {
                    idx_arr.swap(idx1, idx2);
                    val_arr.swap(idx1, idx2);
                }
            }
        }

        // Compare adjacent elements
        for j in 0..8 {
            if (j & 1) == 0 {
                let idx1 = base + j;
                let idx2 = base + j + 1;
                if idx_arr[idx1] > idx_arr[idx2] {
                    idx_arr.swap(idx1, idx2);
                    val_arr.swap(idx1, idx2);
                }
            }
        }
    }

    // Stage 4: Final merge to sort all 16 elements
    // Compare elements 8 apart
    for i in 0..8 {
        let idx1 = i;
        let idx2 = i + 8;
        if idx_arr[idx1] > idx_arr[idx2] {
            idx_arr.swap(idx1, idx2);
            val_arr.swap(idx1, idx2);
        }
    }

    // Compare elements 4 apart
    for i in 0..16 {
        if (i & 4) == 0 {
            let idx2 = i + 4;
            if idx2 < 16 && idx_arr[i] > idx_arr[idx2] {
                idx_arr.swap(i, idx2);
                val_arr.swap(i, idx2);
            }
        }
    }

    // Compare elements 2 apart
    for i in 0..16 {
        if (i & 2) == 0 {
            let idx2 = i + 2;
            if idx2 < 16 && idx_arr[i] > idx_arr[idx2] {
                idx_arr.swap(i, idx2);
                val_arr.swap(i, idx2);
            }
        }
    }

    // Compare adjacent elements
    for i in 0..16 {
        if (i & 1) == 0 {
            let idx2 = i + 1;
            if idx2 < 16 && idx_arr[i] > idx_arr[idx2] {
                idx_arr.swap(i, idx2);
                val_arr.swap(i, idx2);
            }
        }
    }

    // Reload into SIMD registers
    *indices = _mm512_loadu_si512(idx_arr.as_ptr() as *const __m512i);
    *values = _mm512_loadu_ps(val_arr.as_ptr());
}
