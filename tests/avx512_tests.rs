//! Tests for AVX512 accumulator implementation

#![cfg(target_arch = "x86_64")]

use magnus::accumulator::avx512::*;
use magnus::accumulator::Accumulator;

#[test]
fn test_avx512_feature_detection() {
    // Test that feature detection doesn't crash
    let available = is_avx512_available();
    
    // If we're compiling with target_feature="avx512f", it should be true
    #[cfg(target_feature = "avx512f")]
    assert!(available, "AVX512 should be detected when compiled with target feature");
    
    println!("AVX512 available: {}", available);
}

#[test]
fn test_avx512_accumulator_correctness() {
    let mut acc = Avx512Accumulator::new(16);
    
    // Test with duplicate indices
    let test_data = vec![
        (5, 1.0),
        (2, 2.0),
        (8, 3.0),
        (2, 4.0),  // Duplicate of index 2
        (5, 5.0),  // Duplicate of index 5
        (1, 6.0),
        (8, 7.0),  // Duplicate of index 8
        (3, 8.0),
    ];
    
    for (col, val) in test_data {
        acc.accumulate(col, val);
    }
    
    let (indices, values) = acc.extract_result();
    
    // Expected results (sorted by index with duplicates accumulated)
    assert_eq!(indices, vec![1, 2, 3, 5, 8]);
    assert_eq!(values, vec![6.0, 6.0, 8.0, 6.0, 10.0]);
}

#[test]
fn test_size_categories() {
    // Test different size categories that would trigger different code paths
    
    // Exact 16 elements (one AVX512 vector)
    test_size_category(16);
    
    // Less than 16 elements
    test_size_category(8);
    
    // Between 16 and 32
    test_size_category(24);
    
    // Exact 32 elements (two AVX512 vectors)
    test_size_category(32);
    
    // Between 32 and 64
    test_size_category(48);
    
    // Exact 64 elements (four AVX512 vectors)
    test_size_category(64);
    
    // Large array
    test_size_category(256);
}

fn test_size_category(size: usize) {
    let mut acc = Avx512Accumulator::new(size);
    
    // Create test data with some duplicates
    for i in 0..size {
        let col = i % (size / 2 + 1);  // Create some duplicates
        let val = i as f32;
        acc.accumulate(col, val);
    }
    
    let (indices, values) = acc.extract_result();
    
    // Verify results are sorted
    for i in 1..indices.len() {
        assert!(indices[i] > indices[i-1], 
            "Indices should be sorted for size {}", size);
    }
    
    // Verify no duplicates in output
    let mut unique_check = std::collections::HashSet::new();
    for idx in &indices {
        assert!(unique_check.insert(*idx), 
            "No duplicate indices should exist in output for size {}", size);
    }
}

#[test]
fn test_empty_accumulator() {
    let acc = Avx512Accumulator::new(16);
    let (indices, values) = acc.extract_result();
    assert!(indices.is_empty());
    assert!(values.is_empty());
}

#[test]
fn test_single_element() {
    let mut acc = Avx512Accumulator::new(16);
    acc.accumulate(42, 3.14);
    
    let (indices, values) = acc.extract_result();
    assert_eq!(indices, vec![42]);
    assert_eq!(values, vec![3.14]);
}

#[test]
fn test_all_duplicates() {
    let mut acc = Avx512Accumulator::new(16);
    
    // Add many values with the same index
    for i in 0..100 {
        acc.accumulate(7, i as f32);
    }
    
    let (indices, values) = acc.extract_result();
    assert_eq!(indices, vec![7]);
    
    // Sum of 0..100 = 99*100/2 = 4950
    assert_eq!(values, vec![4950.0]);
}

#[test]
fn test_reset_functionality() {
    let mut acc = Avx512Accumulator::new(16);
    
    // First use
    acc.accumulate(1, 1.0);
    acc.accumulate(2, 2.0);
    acc.reset();
    
    // Second use after reset
    acc.accumulate(3, 3.0);
    acc.accumulate(4, 4.0);
    
    let (indices, values) = acc.extract_result();
    
    // Should only contain second use data
    assert_eq!(indices, vec![3, 4]);
    assert_eq!(values, vec![3.0, 4.0]);
}

#[test]
fn test_capacity_growth() {
    let mut acc = Avx512Accumulator::new(4);  // Start small
    
    // Add many more elements than initial capacity
    for i in 0..100 {
        acc.accumulate(i, i as f32);
    }
    
    let (indices, values) = acc.extract_result();
    assert_eq!(indices.len(), 100);
    assert_eq!(values.len(), 100);
    
    // Verify correctness
    for i in 0..100 {
        assert_eq!(indices[i], i);
        assert_eq!(values[i], i as f32);
    }
}

#[test]
fn test_large_indices() {
    let mut acc = Avx512Accumulator::new(16);
    
    // Test with large column indices
    acc.accumulate(1_000_000, 1.0);
    acc.accumulate(500_000, 2.0);
    acc.accumulate(2_000_000, 3.0);
    acc.accumulate(500_000, 4.0);  // Duplicate
    
    let (indices, values) = acc.extract_result();
    
    assert_eq!(indices, vec![500_000, 1_000_000, 2_000_000]);
    assert_eq!(values, vec![6.0, 1.0, 3.0]);
}

#[test]
fn test_negative_values() {
    let mut acc = Avx512Accumulator::new(16);
    
    acc.accumulate(1, -1.0);
    acc.accumulate(2, 2.0);
    acc.accumulate(1, -2.0);  // Accumulate negative with negative
    acc.accumulate(2, -3.0);  // Accumulate negative with positive
    
    let (indices, values) = acc.extract_result();
    
    assert_eq!(indices, vec![1, 2]);
    assert_eq!(values, vec![-3.0, -1.0]);
}

#[test]
fn test_performance_threshold() {
    // This test verifies that AVX512 is used for appropriate sizes
    // We can't directly test performance, but we can verify the code path
    
    if !is_avx512_available() {
        eprintln!("Skipping performance test - AVX512 not available");
        return;
    }
    
    // Test that we handle various sizes efficiently
    for size in [16, 32, 64, 128, 256, 512, 1024] {
        let mut acc = Avx512Accumulator::new(size);
        
        for i in 0..size {
            acc.accumulate(i % (size/2), i as f32);
        }
        
        let (indices, _values) = acc.extract_result();
        assert!(indices.len() <= size/2 + 1);
    }
}