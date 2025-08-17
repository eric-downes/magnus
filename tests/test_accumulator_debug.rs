//! Debug test for accumulator

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;
use std::collections::HashMap;

#[test]
fn debug_accumulator_correctness() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Test case that matches the failing test
    let n = 50_000;
    let mut indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    
    for i in 0..n {
        indices.push((i % 1000) as usize); // 0-999 repeating
        values.push((i as f32) * 0.1);
    }
    
    println!("Input: {} elements, expecting 1000 unique indices", n);
    
    // Count expected occurrences
    let mut expected_counts = HashMap::new();
    for &idx in &indices {
        *expected_counts.entry(idx).or_insert(0) += 1;
    }
    println!("Expected unique count: {}", expected_counts.len());
    
    // Run accumulator
    let (result_indices, result_values) = acc.sort_and_accumulate(&indices, &values);
    
    println!("Result: {} unique indices", result_indices.len());
    
    // Check first few and last few results
    if result_indices.len() > 0 {
        println!("First 5 indices: {:?}", &result_indices[..5.min(result_indices.len())]);
        println!("First 5 values: {:?}", &result_values[..5.min(result_values.len())]);
        
        if result_indices.len() > 5 {
            let start = result_indices.len() - 5;
            println!("Last 5 indices: {:?}", &result_indices[start..]);
            println!("Last 5 values: {:?}", &result_values[start..]);
        }
    }
    
    // Check for unexpected indices
    for &idx in &result_indices {
        if idx >= 1000 {
            println!("ERROR: Found index {} which is >= 1000", idx);
        }
    }
    
    // Check for duplicates in output
    let mut seen = HashMap::new();
    for (i, &idx) in result_indices.iter().enumerate() {
        if let Some(prev) = seen.insert(idx, i) {
            println!("ERROR: Duplicate index {} at positions {} and {}", idx, prev, i);
        }
    }
    
    assert_eq!(result_indices.len(), 1000, "Should have exactly 1000 unique indices");
}