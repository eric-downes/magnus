//! Test exact case that's failing

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;

#[test]
fn test_exact_failing_case() {
    if !MetalAccumulator::is_available() {
        println!("Metal not available");
        return;
    }
    
    let acc = MetalAccumulator::new().unwrap();
    
    // Exact same setup as failing test
    let n = 50_000;
    let mut indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    
    for i in 0..n {
        indices.push((i % 1000) as usize); // Many duplicates
        values.push((i as f32) * 0.1);
    }
    
    let (sorted_indices, accumulated_values) = 
        acc.sort_and_accumulate(&indices, &values);
    
    println!("Result count: {}", sorted_indices.len());
    
    // Check index 0
    println!("\nChecking index 0:");
    println!("Result index: {}", sorted_indices[0]);
    println!("Result value: {}", accumulated_values[0]);
    
    // Calculate expected value for index 0
    // Index 0 appears at positions: 0, 1000, 2000, ..., 49000 (50 times)
    // Values at those positions: 0*0.1, 1000*0.1, 2000*0.1, ..., 49000*0.1
    // Sum = (0 + 1000 + 2000 + ... + 49000) * 0.1
    let expected_sum: f32 = (0..50).map(|k| (k * 1000) as f32 * 0.1).sum();
    println!("Expected value: {}", expected_sum);
    
    // Manual calculation
    let manual_sum = (0 + 1000 + 2000 + 3000 + 4000 + 5000 + 6000 + 7000 + 8000 + 9000 +
                     10000 + 11000 + 12000 + 13000 + 14000 + 15000 + 16000 + 17000 + 18000 + 19000 +
                     20000 + 21000 + 22000 + 23000 + 24000 + 25000 + 26000 + 27000 + 28000 + 29000 +
                     30000 + 31000 + 32000 + 33000 + 34000 + 35000 + 36000 + 37000 + 38000 + 39000 +
                     40000 + 41000 + 42000 + 43000 + 44000 + 45000 + 46000 + 47000 + 48000 + 49000) as f32 * 0.1;
    println!("Manual calculation: {}", manual_sum);
    
    // Check a few more indices
    println!("\nFirst 5 results:");
    for i in 0..5.min(sorted_indices.len()) {
        let expected: f32 = (0..50).map(|k| (sorted_indices[i] + k * 1000) as f32 * 0.1).sum();
        println!("Index {}: value={:.2}, expected={:.2}, diff={:.2}", 
                 sorted_indices[i], accumulated_values[i], expected, 
                 (accumulated_values[i] - expected).abs());
    }
    
    // The test expects exactly 1000 unique indices
    assert_eq!(sorted_indices.len(), 1000, "Should have exactly 1000 unique indices");
    
    // And the first value should match
    assert!((accumulated_values[0] - expected_sum).abs() < 1e-3,
            "Value mismatch for index 0: expected {}, got {}", 
            expected_sum, accumulated_values[0]);
}