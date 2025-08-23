//! Accumulator for sorted (index, value) streams.
//!
//! Implements the accumulator described in the project spec: given sorted
//! `indices: &[u32]` and corresponding `values: &[f32]`, produce unique indices
//! and the sum of values for each run of equal indices. Numerically stable via
//! Neumaier (Kahan–Babuška) compensated summation per segment.
//!
//! Complexity: O(n) time, O(u) space where `u` is number of unique indices.
//! Assumes `indices` are sorted ascending. Handles n == 0..=10^7.

#![forbid(unsafe_code)]

use std::fmt;

/// Error type for accumulation.
#[derive(Debug, PartialEq, Eq)]
pub enum AccumulateError {
    /// `indices.len() != values.len()`
    LengthMismatch { indices: usize, values: usize },
}

impl fmt::Display for AccumulateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccumulateError::LengthMismatch { indices, values } => write!(
                f,
                "indices and values lengths differ: {} vs {}",
                indices, values
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AccumulateError {}

/// Accumulate duplicates in a sorted (index, value) stream.
///
/// Returns newly-allocated output vectors `(unique_indices, accumulated_values)`.
pub fn accumulate_sorted(
    indices: &[u32],
    values: &[f32],
) -> Result<(Vec<u32>, Vec<f32>), AccumulateError> {
    let mut out_idx = Vec::new();
    let mut out_val = Vec::new();
    let _count = accumulate_sorted_into(indices, values, &mut out_idx, &mut out_val)?;
    Ok((out_idx, out_val))
}

/// Accumulate duplicates writing directly into the provided output buffers.
///
/// * The output buffers are cleared first and may be reused across calls to avoid
///   reallocations. Their capacity will be grown up to `indices.len()` when needed.
/// * Returns the number of unique entries written.
pub fn accumulate_sorted_into(
    indices: &[u32],
    values: &[f32],
    out_indices: &mut Vec<u32>,
    out_values: &mut Vec<f32>,
) -> Result<usize, AccumulateError> {
    if indices.len() != values.len() {
        return Err(AccumulateError::LengthMismatch { indices: indices.len(), values: values.len() });
    }

    out_indices.clear();
    out_values.clear();
    if indices.is_empty() {
        return Ok(0);
    }

    // Pre-reserve up to worst-case size (no duplicates)
    out_indices.reserve(indices.len());
    out_values.reserve(values.len());

    // Current run state
    let mut cur_idx = indices[0];
    let mut sum = 0.0_f32;      // running sum
    let mut c = 0.0_f32;        // compensation term (Neumaier)

    // Helper: compensate and add x into (sum, c)
    #[inline]
    fn kbn_add(sum: &mut f32, c: &mut f32, x: f32) {
        let t = *sum + x;
        // Neumaier's variant for improved robustness
        if sum.abs() >= x.abs() {
            *c += (*sum - t) + x;
        } else {
            *c += (x - t) + *sum;
        }
        *sum = t;
    }

    // Prime the first element
    kbn_add(&mut sum, &mut c, values[0]);

    for i in 1..indices.len() {
        let idx = indices[i];
        let val = values[i];
        if idx == cur_idx {
            kbn_add(&mut sum, &mut c, val);
        } else {
            out_indices.push(cur_idx);
            out_values.push(sum + c);
            cur_idx = idx;
            sum = 0.0;
            c = 0.0;
            kbn_add(&mut sum, &mut c, val);
        }
    }

    // Flush final group
    out_indices.push(cur_idx);
    out_values.push(sum + c);

    Ok(out_indices.len())
}

// -------------------- Tests --------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, rtol: f32) {
        let denom = b.abs().max(1.0);
        assert!((a - b).abs() <= rtol * denom, "{} !~= {} (rtol={})", a, b, rtol);
    }

    #[test]
    fn empty_input() {
        let (idx, val) = accumulate_sorted(&[], &[]).unwrap();
        assert!(idx.is_empty() && val.is_empty());
    }

    #[test]
    fn single_element() {
        let (idx, val) = accumulate_sorted(&[7], &[3.5]).unwrap();
        assert_eq!(idx, vec![7]);
        assert_eq!(val, vec![3.5]);
    }

    #[test]
    fn basic_functionality() {
        let indices = [0,0,1,1,1,2,2,3,3,3,3];
        let values  = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0];
        let (uidx, uval) = accumulate_sorted(&indices, &values).unwrap();
        assert_eq!(uidx, vec![0,1,2,3]);
        let expected = [3.0,12.0,13.0,38.0];
        for (a,b) in uval.iter().zip(expected.iter()) { assert_close(*a, *b, 1e-6); }
    }

    #[test]
    fn no_duplicates() {
        let indices = [0,1,2,3,4];
        let values  = [1.0,2.0,3.0,4.0,5.0];
        let (uidx, uval) = accumulate_sorted(&indices, &values).unwrap();
        assert_eq!(uidx, indices);
        for (a,b) in uval.iter().zip(values.iter()) { assert_close(*a, *b, 1e-6); }
    }

    #[test]
    fn all_duplicates() {
        let indices = [42,42,42,42];
        let values  = [1.0,2.0,3.0,4.0];
        let (uidx, uval) = accumulate_sorted(&indices, &values).unwrap();
        assert_eq!(uidx, vec![42]);
        assert_close(uval[0], 10.0, 1e-6);
    }

    #[test]
    fn large_pattern() {
        // 50,000 elements where index = i % 1000, value = i * 0.1, already in sorted order
        let mut indices = Vec::with_capacity(50_000);
        let mut values  = Vec::with_capacity(50_000);
        for r in 0u32..1000 {
            for k in 0u32..50 {
                let i = r + 1000*k; // already sorted by r then k
                indices.push(r);
                values.push(i as f32 * 0.1);
            }
        }

        let (u_idx, u_val) = accumulate_sorted(&indices, &values).unwrap();
        assert_eq!(u_idx.len(), 1000);
        assert_eq!(u_idx[0], 0);
        assert_eq!(*u_idx.last().unwrap(), 999);
        for r in 0..1000 {
            // Expected sum: sum_{k=0..49} (r + 1000k)*0.1 = 5*r + 122_500
            let expected = 5.0 * r as f32 + 122_500.0;
            assert_close(u_val[r], expected, 1e-5);
        }
    }

    #[test]
    fn powers_of_two_boundaries() {
        let sizes = [1023, 1024, 1025, 16383, 16384, 16385];
        for &n in &sizes {
            // Build a deterministic pattern of runs with lengths alternating among {1,3}
            let mut indices = Vec::with_capacity(n);
            let mut values  = Vec::with_capacity(n);
            let mut cur = 0u32;
            while indices.len() < n {
                let run = if indices.len() % 7 == 0 { 1 } else { 3 };
                let mut s = 0.0_f32;
                for j in 0..run {
                    if indices.len() == n { break; }
                    indices.push(cur);
                    let v = (j as f32) + 1.0;
                    values.push(v);
                    s += v;
                }
                cur += 1;
            }

            // Reference accumulation (simple scan without compensation)
            let mut exp_idx = Vec::new();
            let mut exp_val = Vec::new();
            let mut i = 0usize;
            while i < indices.len() {
                let idx = indices[i];
                let mut s = 0.0f32;
                while i < indices.len() && indices[i] == idx {
                    s += values[i];
                    i += 1;
                }
                exp_idx.push(idx);
                exp_val.push(s);
            }

            let (u_idx, u_val) = accumulate_sorted(&indices, &values).unwrap();
            assert_eq!(u_idx, exp_idx);
            for (a,b) in u_val.iter().zip(exp_val.iter()) { assert_close(*a, *b, 1e-5); }
        }
    }
}

