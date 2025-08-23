// Cargo.toml: edition = "2021"
// No deps. Requires AVX-512F at runtime.

#![allow(non_camel_case_types)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512_bitonic_fixed {
    use std::arch::x86_64::*;

    // ---------- indices: partner = i ^ j ----------
    #[inline(always)] unsafe fn idx_xor_1() -> __m512i {
        // _mm512_set_epi32 takes (e15..e0); last arg is lane 0
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

    // ---------- per-(k, j) "want max" masks (lane_id bit-j XOR bit-k) ----------
    // (bit 0 = lane 0 / LSB). These ensure per-pair complement.
    const K2_J1:  __mmask16 = 0x6666;

    const K4_J2:  __mmask16 = 0x3C3C;
    const K4_J1:  __mmask16 = 0x5A5A;

    const K8_J4:  __mmask16 = 0x0FF0;
    const K8_J2:  __mmask16 = 0x33CC;
    const K8_J1:  __mmask16 = 0x55AA;

    const K16_J8: __mmask16 = 0xFF00;
    const K16_J4: __mmask16 = 0xF0F0;
    const K16_J2: __mmask16 = 0xCCCC;
    const K16_J1: __mmask16 = 0xAAAA;

    #[inline(always)]
    unsafe fn step16_ps(v: __m512, idx: __m512i, want_max: __mmask16) -> __m512 {
        let w      = _mm512_permutexvar_ps(idx, v);
        // Ordered, non-signaling compare: true where v > w (NaNs => false)
        let swap: __mmask16 = _mm512_cmp_ps_mask(v, w, _CMP_GT_OQ);
        let take_w: __mmask16 = swap ^ want_max;       // XOR to select min/max per lane
        _mm512_mask_blend_ps(take_w, v, w)             // take partner where requested
    }

    /// Sort one __m512 (16 f32) ascending using a correct bitonic network.
    /// No scalar fallback inside the network; entirely SIMD.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn bitonic_sort_16f32_ps(mut v: __m512) -> __m512 {
        // k = 2; j = 1
        v = step16_ps(v, idx_xor_1(), K2_J1);

        // k = 4; j = 2,1
        v = step16_ps(v, idx_xor_2(), K4_J2);
        v = step16_ps(v, idx_xor_1(), K4_J1);

        // k = 8; j = 4,2,1
        v = step16_ps(v, idx_xor_4(), K8_J4);
        v = step16_ps(v, idx_xor_2(), K8_J2);
        v = step16_ps(v, idx_xor_1(), K8_J1);

        // k = 16; j = 8,4,2,1
        v = step16_ps(v, idx_xor_8(), K16_J8);
        v = step16_ps(v, idx_xor_4(), K16_J4);
        v = step16_ps(v, idx_xor_2(), K16_J2);
        v = step16_ps(v, idx_xor_1(), K16_J1);
        v
    }

    /// Public wrapper: sorts &mut [f32] of length 16 in-place with AVX-512.
    /// (Falls back to scalar only if AVX-512F is not available or len != 16.)
    pub fn sort16_in_place_f32(xs: &mut [f32]) {
        if xs.len() != 16 || !std::is_x86_feature_detected!("avx512f") {
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
    use super::avx512_bitonic_fixed::sort16_in_place_f32;

    #[test]
    fn sorts_and_preserves_duplicates() {
        let mut v = [
            3.0, 3.0, 2.0, 5.0, 5.0, 1.0, 4.0, 4.0,
            0.0, 2.0, 2.0, 1.0, 6.0, 6.0, 6.0, 6.0,
        ];
        sort16_in_place_f32(&mut v);
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(v.iter().filter(|&&x| x == 6.0).count(), 4);
    }
}
