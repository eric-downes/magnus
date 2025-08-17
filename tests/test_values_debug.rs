//! Debug test for values

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;

#[test]
fn debug_accumulation_values() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Simple test with known values
    let indices = vec![0, 0, 0, 1, 1, 2];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    println!("Simple test:");
    println!("Input indices: {:?}", indices);
    println!("Input values: {:?}", values);
    
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("Result indices: {:?}", result_indices);
    println!("Result values: {:?}", result_values);
    println!("Expected values: [6.0, 9.0, 6.0]");
    
    assert_eq!(result_indices, vec![0, 1, 2]);
    assert_eq!(result_values, vec![6.0, 9.0, 6.0]);
    
    // Now test the pattern from the failing test
    println!("\nPattern test:");
    let n = 100; // Smaller for debugging
    let mut indices = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..n {
        indices.push((i % 10) as usize); // 0-9 repeating
        values.push((i as f32) * 0.1);
    }
    
    // What do we expect for index 0?
    // It appears at positions: 0, 10, 20, ..., 90
    // Values are: 0*0.1, 10*0.1, 20*0.1, ..., 90*0.1 = 0, 1, 2, ..., 9
    // Sum = 0 + 1 + 2 + ... + 9 = 45
    
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("For {} elements with 10 unique indices:", n);
    println!("Result count: {}", result_indices.len());
    println!("First result: index={}, value={}", result_indices[0], result_values[0]);
    println!("Expected for index 0: value=45.0");
    
    let expected_0: f32 = (0..10).map(|k| (k * 10) as f32 * 0.1).sum();
    println!("Calculated expected: {}", expected_0);
    
    assert!((result_values[0] - expected_0).abs() < 1e-5);
}