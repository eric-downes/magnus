//! Compare our Metal implementation against the reference implementation

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;

// Reference implementation from the other engineer
fn accumulate_sorted_reference(
    indices: &[u32],
    values: &[f32],
) -> (Vec<u32>, Vec<f32>) {
    let mut out_indices = Vec::new();
    let mut out_values = Vec::new();
    
    if indices.is_empty() {
        return (out_indices, out_values);
    }
    
    // Pre-reserve up to worst-case size
    out_indices.reserve(indices.len());
    out_values.reserve(values.len());
    
    // Current run state
    let mut cur_idx = indices[0];
    let mut sum = 0.0_f32;
    let mut c = 0.0_f32;  // compensation term (Neumaier)
    
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
    
    (out_indices, out_values)
}

#[test]
fn compare_metal_vs_reference() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Test case that fails in Metal
    let n = 50_000;
    let mut indices_unsorted = Vec::with_capacity(n);
    let mut values_unsorted = Vec::with_capacity(n);
    
    // Create unsorted data (like our failing test)
    for i in 0..n {
        indices_unsorted.push((i % 1000) as usize);
        values_unsorted.push((i as f32) * 0.1);
    }
    
    // Run Metal implementation
    let (metal_indices, metal_values) = 
        acc.sort_and_accumulate(&indices_unsorted, &values_unsorted);
    
    // Create sorted version for reference implementation
    let mut pairs: Vec<(usize, f32)> = indices_unsorted.iter().cloned()
        .zip(values_unsorted.iter().cloned())
        .collect();
    pairs.sort_by_key(|p| p.0);
    
    let sorted_indices: Vec<u32> = pairs.iter().map(|p| p.0 as u32).collect();
    let sorted_values: Vec<f32> = pairs.iter().map(|p| p.1).collect();
    
    // Run reference implementation
    let (ref_indices, ref_values) = 
        accumulate_sorted_reference(&sorted_indices, &sorted_values);
    
    println!("Metal: {} unique indices", metal_indices.len());
    println!("Reference: {} unique indices", ref_indices.len());
    
    // Compare first few values
    println!("\nFirst 5 comparisons:");
    for i in 0..5.min(metal_indices.len()).min(ref_indices.len()) {
        println!("Index {}: Metal={:.2}, Reference={:.2}, Diff={:.2}",
                 i, metal_values[i], ref_values[i], 
                 (metal_values[i] - ref_values[i]).abs());
    }
    
    // Check specific failure case
    println!("\nIndex 0 comparison:");
    println!("Metal value: {}", metal_values[0]);
    println!("Reference value: {}", ref_values[0]);
    println!("Expected value: 122500.0");
    
    // Don't assert - just show the differences
    if metal_indices.len() != ref_indices.len() {
        println!("\nERROR: Different number of unique indices!");
    }
    
    let mut mismatches = 0;
    for i in 0..metal_indices.len().min(ref_indices.len()) {
        if metal_indices[i] != ref_indices[i] as usize {
            println!("Index mismatch at position {}: Metal={}, Reference={}",
                     i, metal_indices[i], ref_indices[i]);
            mismatches += 1;
        }
        let diff = (metal_values[i] - ref_values[i]).abs();
        if diff > 1e-3 {
            if mismatches < 10 {  // Only print first 10
                println!("Value mismatch at index {}: Metal={}, Reference={}, Diff={}",
                         metal_indices[i], metal_values[i], ref_values[i], diff);
            }
            mismatches += 1;
        }
    }
    
    if mismatches > 0 {
        println!("\nTotal mismatches: {}", mismatches);
    } else {
        println!("\nAll values match!");
    }
}

#[test]
fn test_reference_with_small_case() {
    // Test the reference implementation with a small case
    let indices = vec![0u32, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    
    let (result_indices, result_values) = 
        accumulate_sorted_reference(&indices, &values);
    
    println!("Indices: {:?}", result_indices);
    println!("Values: {:?}", result_values);
    
    assert_eq!(result_indices, vec![0, 1, 2, 3]);
    assert_eq!(result_values, vec![3.0, 12.0, 13.0, 38.0]);
}

#[test]
fn test_reference_correctness() {
    // Test the large pattern to verify reference implementation is correct
    let n = 50_000;
    let mut indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    
    // Create data in sorted order (like the other engineer's test)
    for r in 0u32..1000 {
        for k in 0u32..50 {
            let i = r + 1000 * k;
            indices.push(r);
            values.push(i as f32 * 0.1);
        }
    }
    
    let (u_idx, u_val) = accumulate_sorted_reference(&indices, &values);
    
    assert_eq!(u_idx.len(), 1000);
    
    // Check index 0 specifically
    let expected_0 = 122_500.0;  // sum of (0 + 1000 + 2000 + ... + 49000) * 0.1
    println!("Index 0: got {}, expected {}", u_val[0], expected_0);
    assert!((u_val[0] - expected_0).abs() < 1e-3,
            "Index 0 mismatch: got {}, expected {}", u_val[0], expected_0);
}