// Cargo.toml:
// [package]
// name = "avx512_bitonic_sort"
// edition = "2021"
//
// [dependencies]
// # none

#![allow(non_camel_case_types)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512_bitonic {
    use core::arch::x86_64::*;

    /// One compare-exchange "step" of a bitonic network for 16 f32 lanes.
    /// `idx` pairs each lane i with lane (i ^ j) via a permute;
    /// `desc_mask` selects MAX for lanes that are in "descending" groups (else MIN).
    #[inline(always)]
    unsafe fn step16_ps(v: __m512, idx: __m512i, desc_mask: __mmask16) -> __m512 {
        let w   = _mm512_permutexvar_ps(idx, v);
        let lo  = _mm512_min_ps(v, w);
        let hi  = _mm512_max_ps(v, w);
        // Take HI in lanes that should be descending, LO otherwise.
        _mm512_mask_blend_ps(desc_mask, lo, hi)
    }

    // Prebaked index vectors for i ^ j (j = 1,2,4,8).
    #[inline(always)] unsafe fn idx_xor_1() -> __m512i {
        // args are e15..e0; last arg is lane 0
        _mm512_set_epi32(14,15,12,13,10,11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1)
    }
    #[inline(always)] unsafe fn idx_xor_2() -> __m512i {
        _mm512_set_epi32(13,12,15,14, 9, 8,11,10, 5, 4, 7, 6, 1, 0, 3, 2)
    }
    #[inline(always)] unsafe fn idx_xor_4() -> __m512i {
        _mm512_set_epi32(11,10, 9, 8,15,14,13,12, 3, 2, 1, 0, 7, 6, 5, 4)
    }
    #[inline(always)] unsafe fn idx_xor_8() -> __m512i {
        _mm512_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0,15,14,13,12,11,10, 9, 8)
    }

    // Masks for which lanes are "descending" given k = 2,4,8,16.
    const DESC_K2:  __mmask16 = 0xCCCC; // pattern ..0011 0011 0011 0011
    const DESC_K4:  __mmask16 = 0xF0F0; // pattern ..1111 0000 1111 0000
    const DESC_K8:  __mmask16 = 0xFF00; // pattern ..1111 1111 0000 0000
    const DESC_K16: __mmask16 = 0x0000; // all ascending for the final stage

    /// Sort 16 f32 (one __m512) ascending using a bitonic network.
    #[inline(always)]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn bitonic_sort_16f32_ps(mut v: __m512) -> __m512 {
        // k = 2; j = 1
        v = step16_ps(v, idx_xor_1(), DESC_K2);

        // k = 4; j = 2,1
        v = step16_ps(v, idx_xor_2(), DESC_K4);
        v = step16_ps(v, idx_xor_1(), DESC_K4);

        // k = 8; j = 4,2,1
        v = step16_ps(v, idx_xor_4(), DESC_K8);
        v = step16_ps(v, idx_xor_2(), DESC_K8);
        v = step16_ps(v, idx_xor_1(), DESC_K8);

        // k = 16; j = 8,4,2,1
        v = step16_ps(v, idx_xor_8(), DESC_K16);
        v = step16_ps(v, idx_xor_4(), DESC_K16);
        v = step16_ps(v, idx_xor_2(), DESC_K16);
        v = step16_ps(v, idx_xor_1(), DESC_K16);

        v
    }

    /// Sorts a slice of 16 f32 in place using AVX-512 bitonic network.
    /// Falls back to scalar sort for other lengths or when AVX-512F is unavailable.
    pub fn sort16_in_place_f32(xs: &mut [f32]) {
        if xs.len() != 16 {
            // Fallback (kept simple). If you don't want fallback, return Err instead.
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            return;
        }
        // Runtime feature guard
        if !std::is_x86_feature_detected!("avx512f") {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            return;
        }

        unsafe {
            let p = xs.as_mut_ptr();
            let v = _mm512_loadu_ps(p);
            let s = bitonic_sort_16f32_ps(v);
            _mm512_storeu_ps(p, s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::avx512_bitonic::sort16_in_place_f32;

    #[test]
    fn sorts_16() {
        let mut v = [
            9.0, 3.0, 5.0, 2.0, 7.0, 1.0, 8.0, 6.0,
            0.0, 4.0, 12.0, 11.0, 10.0, 15.0, 14.0, 13.0,
        ];
        sort16_in_place_f32(&mut v);
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
}
