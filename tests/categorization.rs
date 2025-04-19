//! Integration tests for row categorization

use magnus::{analyze_categorization, categorize_rows};
use magnus::{MagnusConfig, RowCategory, SparseMatrixCSR};

/// Creates a test matrix with dense rows
fn create_dense_matrix(rows: usize, cols: usize, nnz_per_row: usize) -> SparseMatrixCSR<f64> {
    assert!(nnz_per_row <= cols, "nnz_per_row cannot exceed cols");

    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx = Vec::with_capacity(rows * nnz_per_row);
    let mut values = Vec::with_capacity(rows * nnz_per_row);

    row_ptr.push(0);

    for i in 0..rows {
        let start_col = i % (cols - nnz_per_row + 1);

        for j in 0..nnz_per_row {
            col_idx.push(start_col + j);
            values.push(1.0);
        }

        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values)
}

/// Creates a sparse diagonal matrix with bandwidth
fn create_banded_matrix(size: usize, bandwidth: usize) -> SparseMatrixCSR<f64> {
    let nnz_per_row = 2 * bandwidth + 1;
    let mut row_ptr = Vec::with_capacity(size + 1);
    let mut col_idx = Vec::with_capacity(size * nnz_per_row);
    let mut values = Vec::with_capacity(size * nnz_per_row);

    row_ptr.push(0);

    for i in 0..size {
        let start_col = if i >= bandwidth { i - bandwidth } else { 0 };
        let end_col = if i + bandwidth < size {
            i + bandwidth + 1
        } else {
            size
        };

        for j in start_col..end_col {
            col_idx.push(j);
            values.push(1.0);
        }

        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(size, size, row_ptr, col_idx, values)
}

#[test]
fn test_small_matrix_categorization() {
    // Create small matrices where all rows should use Sort
    let a = create_dense_matrix(10, 10, 2);
    let b = create_dense_matrix(10, 10, 2);

    let config = MagnusConfig::default();

    // Categorize rows
    let categories = categorize_rows(&a, &b, &config);

    // All should be categorized as Sort
    assert_eq!(categories.len(), 10);
    assert!(categories.iter().all(|&c| c == RowCategory::Sort));

    // Verify with summary
    let summary = analyze_categorization(&a, &b, &config);
    assert_eq!(summary.total_rows, 10);
    assert_eq!(summary.sort_count, 10);
    assert_eq!(summary.dense_count, 0);
    assert_eq!(summary.fine_level_count, 0);
    assert_eq!(summary.coarse_level_count, 0);
}

#[test]
fn test_mixed_categorization() {
    // Create matrices that will produce different categorizations

    // Customize config to make testing easier
    let mut config = MagnusConfig::default();
    config.dense_accum_threshold = 20; // Small threshold to force different categories

    // Using a small L2 cache size for testing
    config.system_params.l2_cache_size = 1000; // 1KB, very small for testing

    // Matrix A: different rows have different nnz patterns
    // - Row 0: 1 nnz (small, will use Sort)
    // - Row 1: 5 nnz (medium, will exceed Sort threshold)
    // - Row 2: 20 nnz (large, will need reordering)
    let a = SparseMatrixCSR::new(
        3,
        100,
        vec![0, 1, 6, 26],
        (0..26).collect::<Vec<_>>(), // 0, 1, 2, 3, 4, 5, ..., 25
        vec![1.0; 26],
    );

    // Matrix B: each row has 5 non-zeros
    let b = create_dense_matrix(100, 100, 5);

    // Categorize rows
    let categories = categorize_rows(&a, &b, &config);

    // First row should use Sort (5 intermediate products)
    assert_eq!(categories[0], RowCategory::Sort);

    // Second row should use Dense or higher (25 intermediate products)
    assert_ne!(categories[1], RowCategory::Sort);

    // Third row should use higher category (not Sort)
    assert_ne!(categories[2], RowCategory::Sort);

    // Get the summary
    let summary = analyze_categorization(&a, &b, &config);
    assert_eq!(summary.total_rows, 3);
    assert_eq!(summary.sort_count, 1);
    assert!(summary.dense_count + summary.fine_level_count + summary.coarse_level_count == 2);
}

#[test]
fn test_large_matrix_categorization() {
    // Create larger matrices that will exercise all categorization options

    let size = 100;

    // Create banded matrices with different bandwidths
    let a = create_banded_matrix(size, 5); // 11 non-zeros per row
    let b = create_banded_matrix(size, 10); // 21 non-zeros per row

    // Customize config
    let mut config = MagnusConfig::default();
    config.dense_accum_threshold = 50; // Threshold that some rows will exceed

    // Get categorization
    let categories = categorize_rows(&a, &b, &config);

    // Verify all rows were categorized
    assert_eq!(categories.len(), size);

    // Get summary
    let summary = analyze_categorization(&a, &b, &config);
    assert_eq!(summary.total_rows, size);

    // Print summary percentages
    let (sort_pct, dense_pct, fine_pct, coarse_pct) = summary.percentages();
    println!("Categorization percentages:");
    println!("  Sort: {:.1}%", sort_pct);
    println!("  Dense: {:.1}%", dense_pct);
    println!("  Fine: {:.1}%", fine_pct);
    println!("  Coarse: {:.1}%", coarse_pct);

    // Verify percentages sum to 100%
    assert!((sort_pct + dense_pct + fine_pct + coarse_pct - 100.0).abs() < 0.001);
}

#[test]
fn test_disable_coarse_level() {
    // Create matrices that would need coarse-level reordering
    let a = SparseMatrixCSR::new(
        1,
        5000, // One row with many potential connections
        vec![0, 5000],
        (0..5000).collect(), // Connect to all rows in B
        vec![1.0; 5000],
    );

    let b = SparseMatrixCSR::new(
        5000,
        5000,
        (0..=5000).collect(), // Each row has 1 non-zero
        (0..5000).collect(),
        vec![1.0; 5000],
    );

    // Use a modified config with a smaller L2 cache to ensure coarse level is needed
    let mut config_enabled = MagnusConfig::default();
    config_enabled.system_params.l2_cache_size = 10000; // Very small cache

    let categories_enabled = categorize_rows(&a, &b, &config_enabled);

    // With small L2 cache and large intermediate products, row should use CoarseLevel
    assert_eq!(categories_enabled[0], RowCategory::CoarseLevel);

    // Now with coarse-level disabled but same small cache
    let mut config_disabled = config_enabled.clone();
    config_disabled.enable_coarse_level = false;

    let categories_disabled = categorize_rows(&a, &b, &config_disabled);

    // Row should use FineLevel instead when coarse level is disabled
    assert_eq!(categories_disabled[0], RowCategory::FineLevel);
}
