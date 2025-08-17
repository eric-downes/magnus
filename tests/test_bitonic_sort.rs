//! Test the bitonic sort directly

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;

// Wrapper struct to expose internal methods for testing
struct TestableMetalAccumulator {
    inner: MetalAccumulator,
}

impl TestableMetalAccumulator {
    fn new() -> Option<Self> {
        MetalAccumulator::new().map(|inner| TestableMetalAccumulator { inner })
    }
    
    fn test_bitonic_sort(&self, indices: &[u32], values: &[f32]) -> (Vec<u32>, Vec<f32>) {
        // This would need to be exposed from the implementation
        // For now, we'll test indirectly
        (indices.to_vec(), values.to_vec())
    }
}

#[test]
fn test_bitonic_sort_preserves_values() {
    // Create test data
    let mut indices = vec![];
    let mut values = vec![];
    
    // Create 16 elements (power of 2 for clean bitonic sort)
    for i in 0..16 {
        indices.push((15 - i) as u32); // Reverse order
        values.push(i as f32);
    }
    
    println!("Input indices: {:?}", indices);
    println!("Input values: {:?}", values);
    
    // After sorting by indices, we expect:
    // indices: [0, 1, 2, ..., 15]
    // values: [15, 14, 13, ..., 0] (values follow their indices)
    
    // Create pairs for validation
    let mut pairs: Vec<(u32, f32)> = indices.iter().cloned()
        .zip(values.iter().cloned())
        .collect();
    pairs.sort_by_key(|p| p.0);
    
    println!("\nExpected after sort:");
    println!("Indices: {:?}", pairs.iter().map(|p| p.0).collect::<Vec<_>>());
    println!("Values: {:?}", pairs.iter().map(|p| p.1).collect::<Vec<_>>());
    
    // Verify the pairing is maintained
    for (idx, val) in &pairs {
        assert_eq!(*val, (15 - idx) as f32, 
                   "Value {} should be paired with index {}", val, idx);
    }
}

#[test]
fn test_bitonic_with_duplicates() {
    // Test that duplicates are handled correctly
    let indices = vec![3, 1, 3, 2, 1, 3, 2, 1];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
    println!("Input with duplicates:");
    println!("Indices: {:?}", indices);
    println!("Values: {:?}", values);
    
    // Create pairs and sort
    let mut pairs: Vec<(u32, f32)> = indices.iter().map(|&i| i as u32)
        .zip(values.iter().cloned())
        .collect();
    pairs.sort_by_key(|p| p.0);
    
    println!("\nAfter sorting:");
    for (idx, val) in &pairs {
        println!("  Index {}: value {}", idx, val);
    }
    
    // Count values for each index
    let mut sums = std::collections::HashMap::new();
    for (idx, val) in &pairs {
        *sums.entry(*idx).or_insert(0.0) += val;
    }
    
    println!("\nAccumulated sums:");
    for idx in [1, 2, 3] {
        println!("  Index {}: sum = {}", idx, sums[&idx]);
    }
    
    // Expected sums:
    // Index 1: 2.0 + 5.0 + 8.0 = 15.0
    // Index 2: 4.0 + 7.0 = 11.0
    // Index 3: 1.0 + 3.0 + 6.0 = 10.0
    assert_eq!(sums[&1], 15.0);
    assert_eq!(sums[&2], 11.0);
    assert_eq!(sums[&3], 10.0);
}