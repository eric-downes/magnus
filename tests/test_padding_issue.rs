//! Test if padding is causing the issue

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;

#[test]
fn test_padding_boundary() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Test with exactly a power of 2 (no padding needed)
    let n = 16384; // 2^14
    let mut indices = Vec::new();
    let mut values = Vec::new();
    
    // Use 100 unique indices
    for i in 0..n {
        indices.push((i % 100) as usize);
        values.push((i as f32) * 1.0);
    }
    
    println!("Testing with {} elements (exact power of 2)", n);
    
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("Got {} unique indices", result_indices.len());
    
    // Check index 0
    // It appears at positions: 0, 100, 200, ..., 16300 (164 times)
    // Sum = 0 + 100 + 200 + ... + 16300 = 100 * (0 + 1 + 2 + ... + 163) 
    //     = 100 * 163 * 164 / 2 = 100 * 13366 = 1336600
    let expected_0: f32 = (0..164).map(|k| (k * 100) as f32).sum();
    println!("Index 0: got {}, expected {}", result_values[0], expected_0);
    
    assert_eq!(result_indices.len(), 100);
    assert!((result_values[0] - expected_0).abs() < 1e-1,
            "Value mismatch: expected {}, got {}", expected_0, result_values[0]);
}

#[test]
fn test_one_past_power_of_two() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Test with 16385 (one past power of 2, will pad to 32768)
    let n = 16385;
    let mut indices = Vec::new();
    let mut values = Vec::new();
    
    // Use 100 unique indices
    for i in 0..n {
        indices.push((i % 100) as usize);
        values.push((i as f32) * 1.0);
    }
    
    println!("Testing with {} elements (will pad to 32768)", n);
    
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("Got {} unique indices", result_indices.len());
    
    // Check index 0
    // It appears at positions: 0, 100, 200, ..., 16300, 16400 (165 times)
    let expected_0: f32 = (0..165).map(|k| (k * 100) as f32).sum();
    println!("Index 0: got {}, expected {}", result_values[0], expected_0);
    
    assert_eq!(result_indices.len(), 100);
    assert!((result_values[0] - expected_0).abs() < 1e-1,
            "Value mismatch: expected {}, got {}", expected_0, result_values[0]);
}