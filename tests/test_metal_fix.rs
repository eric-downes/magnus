//! Test the Metal fix with CPU accumulation

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;

#[test]
fn test_metal_with_cpu_accumulation() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Test with the exact failing case
    let n = 50_000;
    let mut indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    
    for i in 0..n {
        indices.push((i % 1000) as usize);
        values.push((i as f32) * 0.1);
    }
    
    println!("Testing {} elements with {} unique indices", n, 1000);
    
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("Got {} unique indices", result_indices.len());
    
    // Print first few results
    for i in 0..5.min(result_indices.len()) {
        println!("Index {}: value={}", result_indices[i], result_values[i]);
    }
    
    // Check the specific expected value for index 0
    let expected_0 = 122_500.0;
    println!("\nIndex 0: got {}, expected {}", result_values[0], expected_0);
    
    assert_eq!(result_indices.len(), 1000, 
               "Expected 1000 unique indices, got {}", result_indices.len());
    
    assert!((result_values[0] - expected_0).abs() < 1e-3,
            "Index 0 value mismatch: got {}, expected {}", 
            result_values[0], expected_0);
    
    println!("Test passed!");
}