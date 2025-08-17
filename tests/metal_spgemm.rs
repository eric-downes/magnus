//! Comprehensive tests for Metal GPU-accelerated SpGEMM implementation
//!
//! These tests validate the Metal kernels for sparse matrix multiplication,
//! including accumulation, sorting, and full SpGEMM operations.

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use magnus::accumulator::metal_impl::MetalAccumulator;
use magnus::accumulator::SimdAccelerator;
use magnus::SparseMatrixCSR;
use magnus::{magnus_spgemm, MagnusConfig};

fn generate_test_matrix(n: usize, density: f64, seed: u64) -> SparseMatrixCSR<f32> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let nnz_per_row = (n as f64 * density) as usize;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n * nnz_per_row);
    let mut values = Vec::with_capacity(n * nnz_per_row);
    
    row_ptr.push(0);
    for i in 0..n {
        // Add random elements to row i
        let mut cols: Vec<usize> = Vec::new();
        for _ in 0..nnz_per_row {
            let col = rng.gen_range(0..n);
            if !cols.contains(&col) {
                cols.push(col);
            }
        }
        cols.sort();
        for col in cols {
            col_idx.push(col);
            values.push(rng.gen::<f32>() * 10.0);
        }
        row_ptr.push(col_idx.len());
    }
    
    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}

mod metal_device {
    use super::*;
    
    #[test]
    fn test_metal_availability() {
        // Test that Metal device can be detected
        let available = MetalAccumulator::is_available();
        println!("Metal GPU available: {}", available);
        
        if available {
            // If available, should be able to create accelerator
            let acc = MetalAccumulator::new();
            assert!(acc.is_some(), "Failed to create Metal accelerator despite availability");
        }
    }
    
    #[test]
    fn test_metal_threshold() {
        // Test threshold logic
        assert!(!MetalAccumulator::should_use_metal(100));
        assert!(!MetalAccumulator::should_use_metal(9_999));
        assert!(MetalAccumulator::should_use_metal(10_000));
        assert!(MetalAccumulator::should_use_metal(100_000));
    }
}

mod accumulation {
    use super::*;
    
    #[test]
    #[ignore] // Ignore until accumulation kernel is complete
    fn test_sort_accumulate_small() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        let acc = MetalAccumulator::new().unwrap();
        
        // Test small array with duplicates
        let indices = vec![3, 1, 3, 2, 1, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let (sorted_indices, accumulated_values) = 
            acc.sort_and_accumulate(&indices, &values);
        
        assert_eq!(sorted_indices, vec![1, 2, 3]);
        assert_eq!(accumulated_values, vec![7.0, 4.0, 10.0]);
    }
    
    #[test]
    #[ignore] // Ignore until accumulation kernel is complete
    fn test_sort_accumulate_large() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        let acc = MetalAccumulator::new().unwrap();
        
        // Generate large array with many duplicates
        let n = 50_000;
        let mut indices = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        
        for i in 0..n {
            indices.push((i % 1000) as usize); // Many duplicates
            values.push((i as f32) * 0.1);
        }
        
        let (sorted_indices, accumulated_values) = 
            acc.sort_and_accumulate(&indices, &values);
        
        // Should have exactly 1000 unique indices
        assert_eq!(sorted_indices.len(), 1000);
        
        // Verify accumulation correctness
        for i in 0..1000 {
            assert_eq!(sorted_indices[i], i);
            // Each index appears 50 times, so sum should be:
            // sum of (i + 1000*k)*0.1 for k=0..49
            let expected_sum: f32 = (0..50).map(|k| (i + 1000 * k) as f32 * 0.1).sum();
            assert!((accumulated_values[i] - expected_sum).abs() < 1e-3,
                    "Mismatch at index {}: expected {}, got {}", 
                    i, expected_sum, accumulated_values[i]);
        }
    }
    
    #[test]
    #[ignore] // Ignore until accumulation kernel is complete
    fn test_accumulate_no_duplicates() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        let acc = MetalAccumulator::new().unwrap();
        
        // Test with no duplicates
        let indices: Vec<usize> = (0..10000).collect();
        let values: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        
        let (sorted_indices, accumulated_values) = 
            acc.sort_and_accumulate(&indices, &values);
        
        assert_eq!(sorted_indices, indices);
        assert_eq!(accumulated_values, values);
    }
    
    #[test]
    #[ignore] // Ignore until accumulation kernel is complete
    fn test_accumulate_all_duplicates() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        let acc = MetalAccumulator::new().unwrap();
        
        // All elements are the same
        let indices = vec![42; 10000];
        let values = vec![1.0; 10000];
        
        let (sorted_indices, accumulated_values) = 
            acc.sort_and_accumulate(&indices, &values);
        
        assert_eq!(sorted_indices, vec![42]);
        assert_eq!(accumulated_values, vec![10000.0]);
    }
}

mod parallel_scan {
    use super::*;
    
    #[test]
    #[ignore] // Ignore until parallel scan is implemented
    fn test_parallel_prefix_sum() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Test parallel prefix sum computation
        let input = vec![1u32; 10000];
        let expected: Vec<u32> = (0..10000).collect();
        
        // This would call the Metal parallel scan kernel
        // let output = metal_parallel_scan(&input);
        // assert_eq!(output, expected);
    }
    
    #[test]
    #[ignore] // Ignore until parallel scan is implemented
    fn test_segmented_scan() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Test segmented scan for identifying unique boundaries
        let indices = vec![1, 1, 1, 2, 2, 3, 3, 3, 3];
        let segments = vec![1, 0, 0, 1, 0, 1, 0, 0, 0]; // 1 marks segment start
        
        // This would identify segment boundaries for accumulation
        // let boundaries = metal_segmented_scan(&indices);
        // assert_eq!(boundaries, segments);
    }
}

mod spgemm_symbolic {
    use super::*;
    
    #[test]
    #[ignore] // Ignore until symbolic phase is implemented
    fn test_count_nnz_per_row() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        let a = generate_test_matrix(100, 0.1, 42);
        let b = generate_test_matrix(100, 0.1, 43);
        
        // This would compute non-zero counts per row
        // let nnz_per_row = metal_count_nnz_per_row(&a, &b);
        
        // Verify against CPU implementation
        // let expected_nnz = cpu_count_nnz_per_row(&a, &b);
        // assert_eq!(nnz_per_row, expected_nnz);
    }
    
    #[test]
    #[ignore] // Ignore until symbolic phase is implemented
    fn test_compute_row_pointers() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        let nnz_per_row = vec![3, 5, 2, 0, 7, 1];
        let expected_row_ptr = vec![0, 3, 8, 10, 10, 17, 18];
        
        // This would compute row pointers via prefix sum
        // let row_ptr = metal_compute_row_pointers(&nnz_per_row);
        // assert_eq!(row_ptr, expected_row_ptr);
    }
}

mod spgemm_numeric {
    use super::*;
    
    #[test]
    #[ignore] // Ignore until numeric phase is implemented
    fn test_spgemm_small_dense() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Small dense matrices
        let a = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 0, 2, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        
        let b = SparseMatrixCSR::new(
            3, 3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![7.0, 8.0, 9.0, 10.0, 11.0],
        );
        
        // Compute with Metal
        // let c_metal = metal_spgemm(&a, &b);
        
        // Compute expected result
        let expected = SparseMatrixCSR::new(
            3, 3,
            vec![0, 3, 5, 8],
            vec![0, 1, 2, 0, 2, 0, 1, 2],
            vec![7.0, 18.0, 8.0, 61.0, 68.0, 60.0, 45.0, 66.0],
        );
        
        // assert_eq!(c_metal, expected);
    }
    
    #[test]
    #[ignore] // Ignore until numeric phase is implemented
    fn test_spgemm_large_sparse() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Large sparse matrices
        let a = generate_test_matrix(1000, 0.01, 42);
        let b = generate_test_matrix(1000, 0.01, 43);
        
        // Compute with Metal
        // let c_metal = metal_spgemm(&a, &b);
        
        // Compute with CPU reference
        let config = MagnusConfig::default();
        let c_cpu = magnus_spgemm(&a, &b, &config);
        
        // Compare results
        // assert_matrices_equal(&c_metal, &c_cpu);
    }
    
    #[test]
    #[ignore] // Ignore until numeric phase is implemented
    fn test_spgemm_identity_matrix() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Create identity matrix
        let n = 100;
        let identity = SparseMatrixCSR::<f32>::identity(n);
        
        // A * I = A
        let a = generate_test_matrix(n, 0.05, 42);
        // let result = metal_spgemm(&a, &identity);
        // assert_matrices_equal(&result, &a);
    }
}

mod memory_management {
    use super::*;
    
    #[test]
    #[ignore] // Ignore until memory management is implemented
    fn test_memory_prediction() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Test memory prediction for different duplicate ratios
        let test_cases = vec![
            (10000, 0.75), // 75% duplicates
            (10000, 0.50), // 50% duplicates
            (10000, 0.25), // 25% duplicates
            (10000, 0.0),  // No duplicates
        ];
        
        for (size, dup_ratio) in test_cases {
            // let predicted = predict_output_size(size, dup_ratio);
            // let expected = (size as f64 * (1.0 - dup_ratio)) as usize;
            // assert!((predicted as f64 - expected as f64).abs() < size as f64 * 0.1);
        }
    }
    
    #[test]
    #[ignore] // Ignore until memory management is implemented
    fn test_adaptive_kernel_selection() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Test that appropriate kernel is selected based on duplicate ratio
        
        // High duplicates -> Hash-based accumulation
        // let kernel = select_accumulation_kernel(0.8);
        // assert_eq!(kernel, AccumulationKernel::HashBased);
        
        // Medium duplicates -> Segmented sort
        // let kernel = select_accumulation_kernel(0.6);
        // assert_eq!(kernel, AccumulationKernel::SegmentedSort);
        
        // Low duplicates -> Streaming output
        // let kernel = select_accumulation_kernel(0.2);
        // assert_eq!(kernel, AccumulationKernel::StreamingOutput);
    }
}

mod performance {
    use super::*;
    use std::time::Instant;
    
    #[test]
    #[ignore] // Manual test for performance comparison
    fn benchmark_metal_vs_cpu() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal benchmark - GPU not available");
            return;
        }
        
        let sizes = vec![10_000, 50_000, 100_000, 500_000];
        
        for size in sizes {
            println!("\n=== Benchmarking size: {} ===", size);
            
            // Generate test data with duplicates
            let mut indices = Vec::with_capacity(size);
            let mut values = Vec::with_capacity(size);
            
            for i in 0..size {
                indices.push((i % (size / 10)) as usize);
                values.push((i as f32) * 0.1);
            }
            
            // Benchmark CPU sort-accumulate
            let start = Instant::now();
            let (cpu_indices, cpu_values) = cpu_sort_accumulate(&indices, &values);
            let cpu_time = start.elapsed();
            println!("CPU time: {:?}", cpu_time);
            
            // Benchmark Metal sort-accumulate
            if let Some(metal_acc) = MetalAccumulator::new() {
                let start = Instant::now();
                let (metal_indices, metal_values) = 
                    metal_acc.sort_and_accumulate(&indices, &values);
                let metal_time = start.elapsed();
                println!("Metal time: {:?}", metal_time);
                
                let speedup = cpu_time.as_secs_f64() / metal_time.as_secs_f64();
                println!("Speedup: {:.2}x", speedup);
                
                // Verify correctness
                assert_eq!(cpu_indices, metal_indices, "Index mismatch");
                assert_eq!(cpu_values.len(), metal_values.len(), "Value count mismatch");
                for (i, (cpu_val, metal_val)) in cpu_values.iter().zip(metal_values.iter()).enumerate() {
                    assert!((cpu_val - metal_val).abs() < 1e-3,
                            "Value mismatch at {}: CPU={}, Metal={}", i, cpu_val, metal_val);
                }
            }
        }
    }
    
    fn cpu_sort_accumulate(indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
        // Simple CPU implementation for comparison
        let mut pairs: Vec<_> = indices.iter().cloned()
            .zip(values.iter().cloned())
            .collect();
        pairs.sort_by_key(|p| p.0);
        
        let mut result_indices = Vec::new();
        let mut result_values = Vec::new();
        
        if pairs.is_empty() {
            return (result_indices, result_values);
        }
        
        let mut current_idx = pairs[0].0;
        let mut current_val = pairs[0].1;
        
        for (idx, val) in pairs.into_iter().skip(1) {
            if idx == current_idx {
                current_val += val;
            } else {
                result_indices.push(current_idx);
                result_values.push(current_val);
                current_idx = idx;
                current_val = val;
            }
        }
        
        result_indices.push(current_idx);
        result_values.push(current_val);
        
        (result_indices, result_values)
    }
}

mod edge_cases {
    use super::*;
    
    #[test]
    #[ignore] // Ignore until implementation is complete
    fn test_empty_matrices() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Empty matrix multiplication
        let a = SparseMatrixCSR::<f32>::zeros(10, 10);
        let b = SparseMatrixCSR::<f32>::zeros(10, 10);
        
        // let c = metal_spgemm(&a, &b);
        // assert_eq!(c.nnz(), 0);
    }
    
    #[test]
    #[ignore] // Ignore until implementation is complete
    fn test_single_element() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Single element matrices
        let a = SparseMatrixCSR::new(1, 1, vec![0, 1], vec![0], vec![2.0]);
        let b = SparseMatrixCSR::new(1, 1, vec![0, 1], vec![0], vec![3.0]);
        
        // let c = metal_spgemm(&a, &b);
        // assert_eq!(c.values[0], 6.0);
    }
    
    #[test]
    #[ignore] // Ignore until implementation is complete
    fn test_power_of_two_padding() {
        if !MetalAccumulator::is_available() {
            println!("Skipping Metal test - GPU not available");
            return;
        }
        
        // Test that non-power-of-2 sizes are handled correctly
        let sizes = vec![1000, 1001, 1023, 1024, 1025];
        
        for n in sizes {
            let indices: Vec<usize> = (0..n).collect();
            let values: Vec<f32> = vec![1.0; n];
            
            let acc = MetalAccumulator::new().unwrap();
            let (sorted_indices, sorted_values) = 
                acc.sort_and_accumulate(&indices, &values);
            
            assert_eq!(sorted_indices.len(), n);
            assert_eq!(sorted_values.len(), n);
        }
    }
}

// Helper function to compare matrices with tolerance
fn assert_matrices_equal(a: &SparseMatrixCSR<f32>, b: &SparseMatrixCSR<f32>) {
    assert_eq!(a.n_rows, b.n_rows, "Row count mismatch");
    assert_eq!(a.n_cols, b.n_cols, "Column count mismatch");
    assert_eq!(a.nnz(), b.nnz(), "Non-zero count mismatch");
    
    for row in 0..a.n_rows {
        let a_start = a.row_ptr[row];
        let a_end = a.row_ptr[row + 1];
        let b_start = b.row_ptr[row];
        let b_end = b.row_ptr[row + 1];
        
        assert_eq!(a_end - a_start, b_end - b_start, 
                   "Row {} has different number of non-zeros", row);
        
        for (a_idx, b_idx) in (a_start..a_end).zip(b_start..b_end) {
            assert_eq!(a.col_idx[a_idx], b.col_idx[b_idx],
                       "Column index mismatch in row {}", row);
            assert!((a.values[a_idx] - b.values[b_idx]).abs() < 1e-5,
                    "Value mismatch in row {}, col {}: {} vs {}",
                    row, a.col_idx[a_idx], a.values[a_idx], b.values[b_idx]);
        }
    }
}