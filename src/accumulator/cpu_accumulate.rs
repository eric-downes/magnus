//! CPU-based accumulator for sorted data
//! 
//! This is a correct reference implementation based on the specification
//! in NEEDED_ACCUMULATOR.md. It uses Neumaier compensated summation for
//! numerical stability.

/// Accumulate sorted indices and values, combining duplicates
/// 
/// # Arguments
/// * `indices` - Sorted array of indices (u32)
/// * `values` - Corresponding values (f32)
/// 
/// # Returns
/// * Tuple of (unique_indices, accumulated_values)
pub fn accumulate_sorted_cpu(indices: &[u32], values: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut out_indices = Vec::new();
    let mut out_values = Vec::new();
    
    if indices.is_empty() {
        return (out_indices, out_values);
    }
    
    // Pre-reserve for worst case (no duplicates)
    out_indices.reserve(indices.len());
    out_values.reserve(values.len());
    
    // Current run state
    let mut cur_idx = indices[0];
    let mut sum = 0.0_f32;
    let mut c = 0.0_f32;  // compensation term for Neumaier summation
    
    // Helper for compensated addition
    #[inline]
    fn neumaier_add(sum: &mut f32, c: &mut f32, x: f32) {
        let t = *sum + x;
        if sum.abs() >= x.abs() {
            *c += (*sum - t) + x;
        } else {
            *c += (x - t) + *sum;
        }
        *sum = t;
    }
    
    // Add first element
    neumaier_add(&mut sum, &mut c, values[0]);
    
    // Process remaining elements
    for i in 1..indices.len() {
        let idx = indices[i];
        let val = values[i];
        
        if idx == cur_idx {
            // Same index - accumulate
            neumaier_add(&mut sum, &mut c, val);
        } else {
            // Different index - output previous and start new
            out_indices.push(cur_idx);
            out_values.push(sum + c);  // Include compensation
            
            // Start new accumulation
            cur_idx = idx;
            sum = 0.0;
            c = 0.0;
            neumaier_add(&mut sum, &mut c, val);
        }
    }
    
    // Output final group
    out_indices.push(cur_idx);
    out_values.push(sum + c);
    
    (out_indices, out_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_empty() {
        let (idx, val) = accumulate_sorted_cpu(&[], &[]);
        assert!(idx.is_empty());
        assert!(val.is_empty());
    }
    
    #[test]
    fn test_single() {
        let (idx, val) = accumulate_sorted_cpu(&[42], &[3.14]);
        assert_eq!(idx, vec![42]);
        assert_eq!(val, vec![3.14]);
    }
    
    #[test]
    fn test_basic() {
        let indices = vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        
        let (idx, val) = accumulate_sorted_cpu(&indices, &values);
        
        assert_eq!(idx, vec![0, 1, 2, 3]);
        assert_eq!(val, vec![3.0, 12.0, 13.0, 38.0]);
    }
    
    #[test]
    fn test_large_pattern() {
        // Test the 50,000 element pattern
        let n = 50_000;
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        // Create sorted data
        for r in 0u32..1000 {
            for k in 0u32..50 {
                indices.push(r);
                values.push((r + 1000 * k) as f32 * 0.1);
            }
        }
        
        let (u_idx, u_val) = accumulate_sorted_cpu(&indices, &values);
        
        assert_eq!(u_idx.len(), 1000);
        
        // Check index 0: sum of (0, 1000, 2000, ..., 49000) * 0.1 = 122500
        let expected_0 = 122_500.0;
        assert!((u_val[0] - expected_0).abs() < 1e-3,
                "Index 0: expected {}, got {}", expected_0, u_val[0]);
    }
}