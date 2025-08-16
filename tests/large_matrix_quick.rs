/// Quick large matrix tests for TDD (run in <10 seconds)
///
/// These tests validate core functionality with moderately large matrices
/// without blocking rapid development cycles.
use magnus::{magnus_spgemm, magnus_spgemm_parallel, MagnusConfig, SparseMatrixCSR};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Generate a quick test matrix with predictable nnz count
fn generate_test_matrix(rows: usize, cols: usize, nnz_per_row: usize) -> SparseMatrixCSR<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0);

    for _ in 0..rows {
        // More predictable: exactly nnz_per_row (with small variation)
        let variation = if nnz_per_row > 10 {
            rng.gen_range(0..=(nnz_per_row / 10))
        } else {
            0
        };
        let row_nnz = (nnz_per_row.saturating_sub(variation / 2) + variation).min(cols);

        let mut row_cols = std::collections::BTreeSet::new();

        // Keep adding until we have row_nnz unique columns
        while row_cols.len() < row_nnz {
            row_cols.insert(rng.gen_range(0..cols));
        }

        for col in row_cols {
            col_idx.push(col);
            values.push(rng.gen_range(0.1..10.0));
        }

        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

/// Quick test with ~100K non-zeros
#[test]
fn test_quick_100k_nnz() {
    let a = generate_test_matrix(2_000, 2_000, 50);
    let b = generate_test_matrix(2_000, 2_000, 50);

    let config = MagnusConfig::default();
    let c = magnus_spgemm(&a, &b, &config);

    // Basic validation
    assert_eq!(c.n_rows, a.n_rows);
    assert_eq!(c.n_cols, b.n_cols);
    assert!(c.col_idx.len() > 0);

    // Check column sorting
    for i in 0..c.n_rows {
        let start = c.row_ptr[i];
        let end = c.row_ptr[i + 1];
        for j in start..end - 1 {
            assert!(c.col_idx[j] < c.col_idx[j + 1]);
        }
    }
}

/// Quick parallel test with ~250K non-zeros
#[test]
fn test_quick_250k_parallel() {
    let a = generate_test_matrix(5_000, 5_000, 50);
    let b = generate_test_matrix(5_000, 5_000, 50);

    let nnz_a = a.col_idx.len();
    let nnz_b = b.col_idx.len();
    assert!(nnz_a > 200_000 && nnz_a < 300_000);
    assert!(nnz_b > 200_000 && nnz_b < 300_000);

    let config = MagnusConfig::default();

    // Test parallel execution
    let c_parallel = magnus_spgemm_parallel(&a, &b, &config);

    // Basic validation
    assert_eq!(c_parallel.n_rows, a.n_rows);
    assert_eq!(c_parallel.n_cols, b.n_cols);
    assert!(c_parallel.col_idx.len() > nnz_a);
}

/// Quick test for sparse matrix (approaching 1M nnz)
#[test]
fn test_quick_near_1m() {
    // 10,000 x 10,000 with ~100 nnz/row ≈ 1M nnz
    let a = generate_test_matrix(10_000, 10_000, 100);

    let nnz = a.col_idx.len();
    // Expect around 1M with ±10% variation
    assert!(nnz > 900_000 && nnz < 1_100_000, "nnz = {}", nnz);

    let config = MagnusConfig::default();

    // Square the matrix
    let c = magnus_spgemm_parallel(&a, &a, &config);

    // Validate
    assert_eq!(c.n_rows, a.n_rows);
    assert_eq!(c.n_cols, a.n_cols);

    // Check fill factor is reasonable (squaring can lead to significant fill)
    let fill_factor = c.col_idx.len() as f64 / nnz as f64;
    assert!(
        fill_factor >= 1.0 && fill_factor < 100.0,
        "fill_factor = {}",
        fill_factor
    );
}

/// Test categorization on varied sizes
#[test]
fn test_quick_categorization() {
    use magnus::analyze_categorization;

    // Small dense
    let small_dense = generate_test_matrix(1_000, 1_000, 200);

    // Large sparse
    let large_sparse = generate_test_matrix(10_000, 10_000, 10);

    let config = MagnusConfig::default();

    // Check categorization works
    let summary1 = analyze_categorization(&small_dense, &small_dense, &config);
    assert_eq!(summary1.total_rows, small_dense.n_rows);

    let summary2 = analyze_categorization(&large_sparse, &large_sparse, &config);
    assert_eq!(summary2.total_rows, large_sparse.n_rows);
}

/// Quick correctness check - serial vs parallel
#[test]
fn test_quick_correctness_parallel() {
    let a = generate_test_matrix(3_000, 3_000, 80);
    let b = generate_test_matrix(3_000, 3_000, 80);

    let config = MagnusConfig::default();

    let c_serial = magnus_spgemm(&a, &b, &config);
    let c_parallel = magnus_spgemm_parallel(&a, &b, &config);

    // Same structure
    assert_eq!(c_serial.n_rows, c_parallel.n_rows);
    assert_eq!(c_serial.n_cols, c_parallel.n_cols);
    assert_eq!(c_serial.row_ptr, c_parallel.row_ptr);

    // Values should match (order within rows might differ)
    for i in 0..c_serial.n_rows {
        let start = c_serial.row_ptr[i];
        let end = c_serial.row_ptr[i + 1];

        if end > start {
            // Collect and sort for comparison
            let mut serial_row: Vec<_> = (start..end)
                .map(|j| (c_serial.col_idx[j], c_serial.values[j]))
                .collect();
            let mut parallel_row: Vec<_> = (start..end)
                .map(|j| (c_parallel.col_idx[j], c_parallel.values[j]))
                .collect();

            serial_row.sort_by_key(|&(col, _)| col);
            parallel_row.sort_by_key(|&(col, _)| col);

            for (s, p) in serial_row.iter().zip(parallel_row.iter()) {
                assert_eq!(s.0, p.0);
                assert!((s.1 - p.1).abs() < 1e-10);
            }
        }
    }
}
