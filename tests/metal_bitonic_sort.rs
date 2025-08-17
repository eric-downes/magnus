//! Comprehensive tests for the Metal bitonic sort implementation
//!
//! These tests verify the correctness of the GPU-accelerated bitonic sort
//! with various input patterns and edge cases.

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;
use rand::prelude::*;
use std::collections::HashMap;

/// Helper to generate random indices with specified characteristics
fn generate_test_indices(n: usize, max_value: usize, duplicates: bool) -> Vec<usize> {
    let mut rng = thread_rng();

    if duplicates {
        // Generate indices with many duplicates
        (0..n)
            .map(|_| rng.gen_range(0..max_value.min(n / 10).max(10)))
            .collect()
    } else {
        // Generate mostly unique indices
        let mut indices: Vec<usize> = (0..n.min(max_value)).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n);
        indices
    }
}

/// Helper to verify sorting and accumulation
fn verify_sorted_and_accumulated(
    original_indices: &[usize],
    original_values: &[f32],
    result_indices: &[usize],
    result_values: &[f32],
) {
    // Build expected result
    let mut expected: HashMap<usize, f32> = HashMap::new();
    for (idx, val) in original_indices.iter().zip(original_values.iter()) {
        *expected.entry(*idx).or_insert(0.0) += val;
    }

    // Convert to sorted vectors
    let mut expected_pairs: Vec<(usize, f32)> = expected.into_iter().collect();
    expected_pairs.sort_by_key(|&(idx, _)| idx);

    // Verify length
    assert_eq!(
        result_indices.len(),
        expected_pairs.len(),
        "Result has wrong number of unique indices"
    );

    // Verify sorted order
    for window in result_indices.windows(2) {
        assert!(
            window[0] < window[1],
            "Result indices not properly sorted: {} >= {}",
            window[0],
            window[1]
        );
    }

    // Verify values match
    for (i, &idx) in result_indices.iter().enumerate() {
        let expected_idx = expected_pairs[i].0;
        let expected_val = expected_pairs[i].1;
        let actual_val = result_values[i];

        assert_eq!(idx, expected_idx, "Index mismatch at position {}", i);
        // Use relative error for large values, absolute for small
        let tolerance = (expected_val.abs() * 1e-5).max(1e-3);
        assert!(
            (actual_val - expected_val).abs() < tolerance,
            "Value mismatch for index {}: expected {}, got {}",
            idx,
            expected_val,
            actual_val
        );
    }
}

#[test]
fn test_metal_empty_input() {
    if let Some(acc) = MetalAccumulator::new() {
        let indices: Vec<usize> = vec![];
        let values: Vec<f32> = vec![];

        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        assert!(result_idx.is_empty());
        assert!(result_val.is_empty());
    } else {
        println!("Metal not available, skipping test");
    }
}

#[test]
fn test_metal_single_element() {
    if let Some(acc) = MetalAccumulator::new() {
        let indices = vec![42];
        let values = vec![3.14];

        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        assert_eq!(result_idx, vec![42]);
        assert_eq!(result_val, vec![3.14]);
    }
}

#[test]
fn test_metal_already_sorted() {
    if let Some(acc) = MetalAccumulator::new() {
        // Test with data that's already sorted
        let indices: Vec<usize> = (0..10000).collect();
        let values: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        verify_sorted_and_accumulated(&indices, &values, &result_idx, &result_val);
    }
}

#[test]
fn test_metal_reverse_sorted() {
    if let Some(acc) = MetalAccumulator::new() {
        // Test with reverse sorted data
        let indices: Vec<usize> = (0..10000).rev().collect();
        let values: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        verify_sorted_and_accumulated(&indices, &values, &result_idx, &result_val);
    }
}

#[test]
fn test_metal_all_same_index() {
    if let Some(acc) = MetalAccumulator::new() {
        // All elements have the same index
        let indices = vec![42; 10000];
        let values: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        assert_eq!(result_idx.len(), 1);
        assert_eq!(result_idx[0], 42);

        let expected_sum: f32 = (0..10000).map(|i| i as f32).sum();
        assert!((result_val[0] - expected_sum).abs() < 1e-3);
    }
}

#[test]
fn test_metal_random_small() {
    if let Some(acc) = MetalAccumulator::new() {
        for _ in 0..10 {
            let indices = generate_test_indices(10000, 1000, true);
            let values: Vec<f32> = (0..10000).map(|i| (i as f32) * 0.1).collect();

            let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

            verify_sorted_and_accumulated(&indices, &values, &result_idx, &result_val);
        }
    }
}

#[test]
fn test_metal_random_large() {
    if let Some(acc) = MetalAccumulator::new() {
        // Test with larger arrays
        let indices = generate_test_indices(50000, 10000, true);
        let values: Vec<f32> = (0..50000).map(|i| (i as f32) * 0.01).collect();

        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        verify_sorted_and_accumulated(&indices, &values, &result_idx, &result_val);
    }
}

#[test]
fn test_metal_power_of_two_sizes() {
    if let Some(acc) = MetalAccumulator::new() {
        // Test exact power-of-two sizes
        for power in [12, 13, 14] {
            let size = 1 << power; // 4096, 8192, 16384
            let indices = generate_test_indices(size, size / 4, true);
            let values: Vec<f32> = (0..size).map(|i| i as f32).collect();

            let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

            verify_sorted_and_accumulated(&indices, &values, &result_idx, &result_val);
        }
    }
}

#[test]
fn test_metal_non_power_of_two_sizes() {
    if let Some(acc) = MetalAccumulator::new() {
        // Test non-power-of-two sizes (requires padding)
        for size in [10001, 15000, 20000, 33333] {
            let indices = generate_test_indices(size, size / 5, true);
            let values: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

            let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

            verify_sorted_and_accumulated(&indices, &values, &result_idx, &result_val);
        }
    }
}

#[test]
fn test_metal_sparse_pattern() {
    if let Some(acc) = MetalAccumulator::new() {
        // Simulate sparse matrix pattern: few unique indices, many duplicates
        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Create clusters of indices
        for cluster in 0..100 {
            let cluster_size = 100;
            for _ in 0..cluster_size {
                indices.push(cluster * 10);
                values.push(cluster as f32);
            }
        }

        // Shuffle to test sorting
        let mut combined: Vec<_> = indices.iter().zip(values.iter()).collect();
        combined.shuffle(&mut thread_rng());

        let indices: Vec<usize> = combined.iter().map(|(&idx, _)| idx).collect();
        let values: Vec<f32> = combined.iter().map(|(_, &val)| val).collect();

        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        verify_sorted_and_accumulated(&indices, &values, &result_idx, &result_val);
    }
}

#[test]
fn test_metal_edge_values() {
    if let Some(acc) = MetalAccumulator::new() {
        // Test with edge case values
        let indices = vec![0, usize::MAX - 1, 1000, usize::MAX - 1, 0];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // This should work with small inputs (uses CPU path)
        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        assert_eq!(result_idx.len(), 3);
        assert_eq!(result_idx[0], 0);
        assert_eq!(result_idx[1], 1000);
        assert_eq!(result_idx[2], usize::MAX - 1);

        assert!((result_val[0] - 6.0).abs() < 1e-5); // 1.0 + 5.0
        assert!((result_val[1] - 3.0).abs() < 1e-5);
        assert!((result_val[2] - 6.0).abs() < 1e-5); // 2.0 + 4.0
    }
}

#[test]
fn test_metal_numerical_stability() {
    if let Some(acc) = MetalAccumulator::new() {
        // Test accumulation with many small values
        let indices: Vec<usize> = (0..20000).map(|_| 0).collect(); // All same index
        let values: Vec<f32> = (0..20000).map(|_| 1e-6).collect();

        let (result_idx, result_val) = acc.sort_and_accumulate(&indices, &values);

        assert_eq!(result_idx.len(), 1);
        assert_eq!(result_idx[0], 0);

        let expected_sum = 20000.0 * 1e-6;
        let relative_error = ((result_val[0] - expected_sum) / expected_sum).abs();
        assert!(
            relative_error < 1e-3,
            "Numerical error too large: expected {}, got {}",
            expected_sum,
            result_val[0]
        );
    }
}

/// Stress test with maximum supported size
#[test]
#[ignore] // This test is expensive, run with --ignored flag
fn test_metal_stress_large() {
    if let Some(acc) = MetalAccumulator::new() {
        let size = 1 << 20; // 1M elements
        let indices = generate_test_indices(size, size / 100, true);
        let values: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

        let (result_idx, _result_val) = acc.sort_and_accumulate(&indices, &values);

        // Basic sanity checks
        assert!(!result_idx.is_empty());
        assert!(result_idx.len() <= size);

        // Verify sorted
        for window in result_idx.windows(2) {
            assert!(window[0] < window[1]);
        }
    }
}
