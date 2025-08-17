//! Test around the Metal threshold

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;

#[test]
fn test_at_threshold() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Test at exactly 10,000 elements (the threshold)
    let n = 10_000;
    let mut indices = Vec::new();
    let mut values = Vec::new();
    
    // Use 100 unique indices, so each appears 100 times
    for i in 0..n {
        indices.push((i % 100) as usize);
        values.push((i as f32) * 1.0); // Use 1.0 for easier calculation
    }
    
    println!("Testing with {} elements, {} unique indices", n, 100);
    
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("Got {} unique indices", result_indices.len());
    
    // Check index 0
    // It appears at positions: 0, 100, 200, ..., 9900 (100 times)
    // Values: 0, 100, 200, ..., 9900
    // Sum = 100 * (0 + 99) * 100 / 2 = 100 * 99 * 50 = 495000
    let expected_0: f32 = (0..100).map(|k| (k * 100) as f32).sum();
    println!("Index 0: got {}, expected {}", result_values[0], expected_0);
    
    assert_eq!(result_indices.len(), 100);
    assert_eq!(result_indices[0], 0);
    assert!((result_values[0] - expected_0).abs() < 1e-3,
            "Value mismatch: expected {}, got {}", expected_0, result_values[0]);
}

#[test]
fn test_just_below_threshold() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Test at 9,999 elements (just below threshold)
    let n = 9_999;
    let mut indices = Vec::new();
    let mut values = Vec::new();
    
    // Use 99 unique indices
    for i in 0..n {
        indices.push((i % 99) as usize);
        values.push((i as f32) * 1.0);
    }
    
    println!("Testing with {} elements, {} unique indices", n, 99);
    
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("Got {} unique indices", result_indices.len());
    
    // Index 0 appears at: 0, 99, 198, ..., 9900 (101 times)
    let expected_0: f32 = (0..101).map(|k| (k * 99) as f32).sum();
    println!("Index 0: got {}, expected {}", result_values[0], expected_0);
    
    assert_eq!(result_indices.len(), 99);
}