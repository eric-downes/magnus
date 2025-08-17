//! Debug what's being accumulated

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;

#[test]
fn debug_what_gets_accumulated() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Small test case to understand the pattern
    // Use 20 elements with 4 unique indices
    let n = 20;
    let mut indices = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..n {
        let idx = (i % 4) as usize;
        let val = (i as f32) * 1.0; // Use 1.0 for easier reading
        indices.push(idx);
        values.push(val);
        println!("Input[{}]: index={}, value={}", i, idx, val);
    }
    
    println!("\nExpected accumulation:");
    println!("Index 0: positions [0,4,8,12,16], values [0,4,8,12,16], sum = 40");
    println!("Index 1: positions [1,5,9,13,17], values [1,5,9,13,17], sum = 45");
    println!("Index 2: positions [2,6,10,14,18], values [2,6,10,14,18], sum = 50");
    println!("Index 3: positions [3,7,11,15,19], values [3,7,11,15,19], sum = 55");
    
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("\nActual results:");
    for i in 0..result_indices.len() {
        println!("Result[{}]: index={}, value={}", i, result_indices[i], result_values[i]);
    }
    
    // Now check if the GPU is using the data correctly
    // Let's manually compute what we expect
    let mut expected = vec![0.0f32; 4];
    for i in 0..n {
        let idx = i % 4;
        expected[idx] += i as f32;
    }
    
    println!("\nExpected computed:");
    for i in 0..4 {
        println!("Index {}: {}", i, expected[i]);
    }
    
    // Check results
    assert_eq!(result_indices.len(), 4);
    for i in 0..4 {
        assert_eq!(result_indices[i], i);
        assert!((result_values[i] - expected[i]).abs() < 1e-5,
                "Mismatch at index {}: expected {}, got {}",
                i, expected[i], result_values[i]);
    }
}