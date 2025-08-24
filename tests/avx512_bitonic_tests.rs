//! Comprehensive tests for AVX512 bitonic sort implementation
//!
//! These tests are designed to catch the common errors that occurred in other
//! architectures' bitonic sort implementations, particularly around:
//! - Direction calculation bugs
//! - Partner thread calculation
//! - Boundary conditions
//! - Padding handling

#![cfg(target_arch = "x86_64")]

use magnus::accumulator::avx512::*;
use magnus::accumulator::Accumulator;

/// Helper function to create test data with known patterns
fn create_test_pairs(indices: Vec<u32>, values: Vec<f32>) -> (Vec<u32>, Vec<f32>) {
    (indices, values)
}

/// Verify that a bitonic sort result is correct
fn verify_sorted(
    indices: &[usize],
    values: &[f32],
    expected_indices: &[usize],
    expected_values: &[f32],
) {
    assert_eq!(
        indices.len(),
        expected_indices.len(),
        "Index length mismatch: got {:?}, expected {:?}",
        indices,
        expected_indices
    );
    assert_eq!(
        values.len(),
        expected_values.len(),
        "Value length mismatch: got {:?}, expected {:?}",
        values,
        expected_values
    );

    for i in 0..indices.len() {
        assert_eq!(
            indices[i], expected_indices[i],
            "Index mismatch at position {}: got {}, expected {}",
            i, indices[i], expected_indices[i]
        );
        assert!(
            (values[i] - expected_values[i]).abs() < 1e-6,
            "Value mismatch at position {}: got {}, expected {}",
            i,
            values[i],
            expected_values[i]
        );
    }
}

#[test]
fn test_bitonic_sort_empty() {
    // Empty input should return empty output
    let mut acc = Avx512Accumulator::new(16);
    let (indices, values) = acc.extract_result();
    assert!(indices.is_empty());
    assert!(values.is_empty());
}

#[test]
fn test_bitonic_sort_single_element() {
    let mut acc = Avx512Accumulator::new(16);
    acc.accumulate(42, 3.14);

    let (indices, values) = acc.extract_result();
    verify_sorted(&indices, &values, &[42usize], &[3.14]);
}

#[test]
fn test_bitonic_sort_two_elements_ascending() {
    let mut acc = Avx512Accumulator::new(16);
    acc.accumulate(10, 1.0);
    acc.accumulate(20, 2.0);

    let (indices, values) = acc.extract_result();
    verify_sorted(&indices, &values, &[10usize, 20], &[1.0, 2.0]);
}

#[test]
fn test_bitonic_sort_two_elements_descending() {
    let mut acc = Avx512Accumulator::new(16);
    acc.accumulate(20, 2.0);
    acc.accumulate(10, 1.0);

    let (indices, values) = acc.extract_result();
    verify_sorted(&indices, &values, &[10usize, 20], &[1.0, 2.0]);
}

#[test]
fn test_bitonic_sort_power_of_two_sizes() {
    // Test exact power-of-two sizes: 2, 4, 8, 16
    for size in [2, 4, 8, 16] {
        let mut acc = Avx512Accumulator::new(size);

        // Add elements in reverse order
        for i in (0..size).rev() {
            acc.accumulate(i, (i + 1) as f32);
        }

        let (indices, values) = acc.extract_result();

        // Verify sorted
        assert_eq!(indices.len(), size);
        for i in 0..size {
            assert_eq!(indices[i], i);
            assert_eq!(values[i], (i + 1) as f32);
        }
    }
}

#[test]
fn test_bitonic_sort_non_power_of_two() {
    // Test sizes that require padding: 3, 5, 7, 9, 11, 13, 15
    for size in [3, 5, 7, 9, 11, 13, 15] {
        let mut acc = Avx512Accumulator::new(size);

        // Add elements in shuffled order
        let mut order: Vec<usize> = (0..size).collect();
        // Simple shuffle for deterministic testing
        for i in 0..size {
            let swap_idx = (i * 7 + 3) % size;
            order.swap(i, swap_idx);
        }

        for &i in &order {
            acc.accumulate(i, (i + 1) as f32);
        }

        let (indices, values) = acc.extract_result();

        // Verify sorted
        assert_eq!(indices.len(), size);
        for i in 0..size {
            assert_eq!(indices[i], i);
            assert_eq!(values[i], (i + 1) as f32);
        }
    }
}

#[test]
fn test_bitonic_sort_duplicates_adjacent() {
    let mut acc = Avx512Accumulator::new(16);

    // Add adjacent duplicates
    acc.accumulate(5, 1.0);
    acc.accumulate(5, 2.0);
    acc.accumulate(10, 3.0);
    acc.accumulate(10, 4.0);

    let (indices, values) = acc.extract_result();
    verify_sorted(&indices, &values, &[5usize, 10], &[3.0, 7.0]);
}

#[test]
fn test_bitonic_sort_duplicates_scattered() {
    let mut acc = Avx512Accumulator::new(16);

    // Add scattered duplicates
    acc.accumulate(10, 1.0);
    acc.accumulate(5, 2.0);
    acc.accumulate(15, 3.0);
    acc.accumulate(5, 4.0);
    acc.accumulate(10, 5.0);
    acc.accumulate(15, 6.0);

    let (indices, values) = acc.extract_result();
    verify_sorted(&indices, &values, &[5usize, 10, 15], &[6.0, 6.0, 9.0]);
}

#[test]
fn test_bitonic_sort_all_same_index() {
    let mut acc = Avx512Accumulator::new(16);

    // All elements have the same index - maximum accumulation
    for i in 0..16 {
        acc.accumulate(42, (i + 1) as f32);
    }

    let (indices, values) = acc.extract_result();
    // Sum of 1..16 = 16*17/2 = 136
    verify_sorted(&indices, &values, &[42usize], &[136.0]);
}

#[test]
fn test_bitonic_sort_alternating_pattern() {
    let mut acc = Avx512Accumulator::new(16);

    // Alternating high-low pattern
    for i in 0..8 {
        acc.accumulate(i * 2, (i + 1) as f32); // Even indices
        acc.accumulate(i * 2 + 1, (i + 10) as f32); // Odd indices
    }

    let (indices, values) = acc.extract_result();

    // Verify all 16 unique indices are present and sorted
    assert_eq!(indices.len(), 16);
    for i in 0..16 {
        assert_eq!(indices[i], i);
    }
}

#[test]
fn test_bitonic_sort_boundary_values() {
    let mut acc = Avx512Accumulator::new(16);

    // Test with boundary values
    acc.accumulate(0, f32::MIN_POSITIVE);
    acc.accumulate(u32::MAX as usize - 1, f32::MAX);
    acc.accumulate(1000000, 1.0);

    let (indices, values) = acc.extract_result();

    assert_eq!(indices[0], 0);
    assert_eq!(indices[1], 1000000);
    assert_eq!(indices[2], u32::MAX as usize - 1);

    assert_eq!(values[0], f32::MIN_POSITIVE);
    assert_eq!(values[1], 1.0);
    assert_eq!(values[2], f32::MAX);
}

#[test]
fn test_bitonic_sort_negative_values() {
    let mut acc = Avx512Accumulator::new(16);

    // Mix of positive and negative values
    acc.accumulate(5, -10.0);
    acc.accumulate(3, 5.0);
    acc.accumulate(7, -2.5);
    acc.accumulate(3, -3.0); // Duplicate that will sum to 2.0
    acc.accumulate(5, 15.0); // Duplicate that will sum to 5.0

    let (indices, values) = acc.extract_result();
    verify_sorted(&indices, &values, &[3usize, 5, 7], &[2.0, 5.0, -2.5]);
}

#[test]
fn test_bitonic_sort_exactly_16_elements() {
    // This is the critical test - exactly 16 elements uses pure AVX512
    let mut acc = Avx512Accumulator::new(16);

    // Add 16 elements in a specific pattern that tests the bitonic sort
    // Use a pattern that would reveal direction calculation bugs
    let test_indices = vec![15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    let test_values: Vec<f32> = test_indices.iter().map(|&i| (i + 1) as f32).collect();

    for i in 0..16 {
        acc.accumulate(test_indices[i] as usize, test_values[i]);
    }

    let (indices, values) = acc.extract_result();

    // Should be sorted 0..15
    assert_eq!(indices.len(), 16);
    for i in 0..16 {
        assert_eq!(
            indices[i], i,
            "Index {} should be {} but got {}",
            i, i, indices[i]
        );
        assert_eq!(
            values[i],
            (i + 1) as f32,
            "Value at {} should be {} but got {}",
            i,
            i + 1,
            values[i]
        );
    }
}

#[test]
fn test_bitonic_sort_stage_boundary_pattern() {
    // Test pattern that would fail with incorrect stage/pass calculations
    let mut acc = Avx512Accumulator::new(16);

    // Create a pattern where pairs need to be compared across stages
    let indices = vec![
        8, 9, 10, 11, 12, 13, 14, 15, // Second half
        0, 1, 2, 3, 4, 5, 6, 7, // First half
    ];

    for (i, &idx) in indices.iter().enumerate() {
        acc.accumulate(idx as usize, (i + 1) as f32);
    }

    let (result_indices, result_values) = acc.extract_result();

    // Verify indices are sorted
    for i in 0..16 {
        assert_eq!(result_indices[i], i);
    }
}

#[test]
fn test_bitonic_sort_with_accumulation_pattern() {
    // Test that accumulation happens correctly during sort
    let mut acc = Avx512Accumulator::new(16);

    // Pattern: each index appears twice with specific values
    for i in 0..8 {
        acc.accumulate(i, (i + 1) as f32);
        acc.accumulate(i, (i + 1) as f32 * 10.0);
    }

    let (indices, values) = acc.extract_result();

    assert_eq!(indices.len(), 8);
    for i in 0..8 {
        assert_eq!(indices[i], i);
        let expected = (i + 1) as f32 * 11.0; // 1x + 10x = 11x
        assert!(
            (values[i] - expected).abs() < 1e-6,
            "Value mismatch at {}: expected {}, got {}",
            i,
            expected,
            values[i]
        );
    }
}

#[test]
fn test_bitonic_direction_bug_pattern() {
    // This specific pattern would fail with the direction calculation bug
    // that was found in the Metal implementation
    let mut acc = Avx512Accumulator::new(16);

    // Add indices in a pattern that tests proper direction handling
    let indices = vec![0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15];
    for &i in &indices {
        acc.accumulate(i as usize, (i + 1) as f32);
    }

    let (result_indices, result_values) = acc.extract_result();

    // Should be perfectly sorted
    for i in 0..16 {
        assert_eq!(
            result_indices[i], i,
            "Direction bug detected: index {} is {} but should be {}",
            i, result_indices[i], i
        );
        assert_eq!(result_values[i], (i + 1) as f32);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore] // Run with --ignored to test performance
    fn benchmark_bitonic_vs_scalar() {
        if !is_avx512_available() {
            eprintln!("Skipping benchmark - AVX512 not available");
            return;
        }

        const ITERATIONS: usize = 10000;

        // Benchmark 16 elements (optimal for AVX512)
        let mut total_time = std::time::Duration::ZERO;

        for _ in 0..ITERATIONS {
            let mut acc = Avx512Accumulator::new(16);

            // Add 16 elements in reverse order
            for i in (0..16).rev() {
                acc.accumulate(i, (i + 1) as f32);
            }

            let start = Instant::now();
            let (_indices, _values) = acc.extract_result();
            total_time += start.elapsed();
        }

        println!(
            "AVX512 bitonic sort (16 elements): {:?} per iteration",
            total_time / ITERATIONS as u32
        );
    }
}
