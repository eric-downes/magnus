use magnus::{SparseMatrixCSR, MagnusConfig, reference_spgemm};
use magnus::{categorize_rows, analyze_categorization};

fn main() {
    println!("MAGNUS: Matrix Algebra for GPU and Multicore Systems");
    println!("Implementation in progress - see README.md for more information");
    
    // Create a simple example matrix
    let a = SparseMatrixCSR::new(
        3, 3,
        vec![0, 2, 3, 5],
        vec![0, 1, 1, 0, 2],
        vec![1, 2, 3, 4, 5],
    );
    
    let b = SparseMatrixCSR::new(
        3, 3,
        vec![0, 2, 3, 4],
        vec![0, 2, 0, 1],
        vec![7, 8, 9, 10],
    );
    
    // Display the matrices
    println!("\nMatrix A:");
    println!("{:?}", a);
    
    println!("\nMatrix B:");
    println!("{:?}", b);
    
    // Display the current configuration
    let config = MagnusConfig::default();
    println!("\nDefault configuration:");
    println!("  Dense accumulator threshold: {}", config.dense_accum_threshold);
    println!("  Coarse level enabled: {}", config.enable_coarse_level);
    println!("  System parameters:");
    println!("    Threads: {}", config.system_params.n_threads);
    println!("    L2 cache size: {} bytes", config.system_params.l2_cache_size);
    
    // Categorize rows
    let categories = categorize_rows(&a, &b, &config);
    let summary = analyze_categorization(&a, &b, &config);
    
    println!("\nRow categorization:");
    for (i, category) in categories.iter().enumerate() {
        println!("  Row {}: {:?}", i, category);
    }
    
    println!("\nCategorization summary:");
    println!("  Total rows: {}", summary.total_rows);
    println!("  Sort-based: {}", summary.sort_count);
    println!("  Dense accumulation: {}", summary.dense_count);
    println!("  Fine-level: {}", summary.fine_level_count);
    println!("  Coarse-level: {}", summary.coarse_level_count);
    
    // Use the reference implementation for now
    println!("\nUsing reference implementation:");
    let result = reference_spgemm(&a, &b);
    println!("{:?}", result);
    
    // Note: MAGNUS implementation not yet complete
    println!("\nMAGNUS implementation in progress...");
}
