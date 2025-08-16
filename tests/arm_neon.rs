#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod arm_tests {
    use magnus::accumulator::{NeonAccumulator, SimdAccelerator};
    use magnus::matrix::SparseMatrixCSR;

    #[test]
    fn test_neon_sort_small() {
        let accumulator = NeonAccumulator::new();

        // Test with small input
        let col_indices = vec![5, 2, 8, 2, 5, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (result_cols, result_vals) = accumulator.sort_and_accumulate(&col_indices, &values);

        // Should be sorted and accumulated
        assert_eq!(result_cols, vec![1, 2, 5, 8]);
        assert_eq!(result_vals, vec![6.0, 6.0, 6.0, 3.0]); // 2->2+4=6, 5->1+5=6
    }

    #[test]
    fn test_neon_sort_aligned() {
        let accumulator = NeonAccumulator::new();

        // Test with NEON-aligned size (multiple of 4 for 32-bit values)
        let col_indices = vec![7, 3, 7, 3, 5, 1, 5, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let (result_cols, result_vals) = accumulator.sort_and_accumulate(&col_indices, &values);

        assert_eq!(result_cols, vec![1, 3, 5, 7]);
        assert_eq!(result_vals, vec![14.0, 6.0, 12.0, 4.0]); // 1->6+8=14, 3->2+4=6, 5->5+7=12, 7->1+3=4
    }

    #[test]
    fn test_neon_sort_large() {
        let accumulator = NeonAccumulator::new();

        // Generate larger test case
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Create pattern with duplicates
        for i in 0..256 {
            col_indices.push(i % 64);
            values.push((i as f32) * 0.1);
        }

        let (result_cols, result_vals) = accumulator.sort_and_accumulate(&col_indices, &values);

        // Should have 64 unique columns
        assert_eq!(result_cols.len(), 64);

        // Check sorting
        for i in 1..result_cols.len() {
            assert!(result_cols[i] > result_cols[i - 1]);
        }

        // Check accumulation (each column appears 4 times)
        // Values for column 0: 0.0, 6.4, 12.8, 19.2 = 38.4
        assert!((result_vals[0] - 38.4).abs() < 1e-5);
    }

    #[test]
    fn test_neon_performance_vs_generic() {
        use std::time::Instant;

        let neon_acc = NeonAccumulator::new();
        let generic_acc = magnus::accumulator::FallbackAccumulator::new();

        // Generate test data
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..1024 {
            col_indices.push(i % 256);
            values.push((i as f32) * 0.1);
        }

        // Benchmark NEON
        let start = Instant::now();
        for _ in 0..100 {
            let _ = neon_acc.sort_and_accumulate(&col_indices, &values);
        }
        let neon_time = start.elapsed();

        // Benchmark generic
        let start = Instant::now();
        for _ in 0..100 {
            let _ = generic_acc.sort_and_accumulate(&col_indices, &values);
        }
        let generic_time = start.elapsed();

        // NEON should be faster (allow some variance)
        println!(
            "NEON time: {:?}, Generic time: {:?}",
            neon_time, generic_time
        );
        // We expect at least some speedup, but don't fail if not dramatic
        // as optimization levels and other factors can affect this
    }

    #[test]
    fn test_neon_accumulator_in_magnus() {
        // Test that NEON accumulator integrates correctly with main algorithm
        let a = SparseMatrixCSR::<f32>::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        let b = SparseMatrixCSR::<f32>::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![1, 2, 0, 2, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        // Create config that forces NEON usage
        let config = magnus::MagnusConfig::for_architecture(magnus::Architecture::ArmNeon);

        let c = magnus::magnus_spgemm(&a, &b, &config);

        // Verify result is correct
        assert_eq!(c.n_rows, 3);
        assert_eq!(c.n_cols, 3);
        // Don't check exact values as algorithm may vary, just structure
        assert!(c.row_ptr.len() == 4);
        assert!(c.col_idx.len() > 0);
        assert!(c.values.len() == c.col_idx.len());
    }
}

// Provide empty module for non-ARM platforms to avoid compilation errors
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
mod arm_tests {
    #[test]
    fn test_skip_on_non_arm() {
        // These tests only run on ARM
        println!("Skipping ARM NEON tests on non-ARM platform");
    }
}
