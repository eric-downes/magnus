//! Parameter space exploration tool for MAGNUS
//!
//! This binary generates test matrices across the parameter space
//! and can be used for systematic performance testing.

use magnus::constants::*;
use magnus::matrix::SparseMatrixCSR;
use magnus::parameter_space::{
    MatrixGenerator, ParameterConfiguration, ParameterSpaceExplorer, PatternMatrixGenerator,
    TestSuiteGenerator,
};
use magnus::suitesparse_integration::{MatrixMarketIO, SuiteSparseStyleGenerator};
use magnus::reference_spgemm;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    println!("MAGNUS Parameter Space Explorer");
    println!("================================\n");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("demo");

    match mode {
        "demo" => run_demo(),
        "generate" => generate_test_suite(),
        "benchmark" => run_benchmark(),
        "patterns" => demonstrate_patterns(),
        "suitesparse" => demonstrate_suitesparse_style(),
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("Usage: explore_parameters <mode>");
    println!();
    println!("Modes:");
    println!("  demo        - Run a demonstration of parameter space exploration");
    println!("  generate    - Generate a complete test suite");
    println!("  benchmark   - Run benchmarks across parameter space");
    println!("  patterns    - Demonstrate pattern-based matrix generation");
    println!("  suitesparse - Generate SuiteSparse-style test matrices");
}

fn run_demo() {
    println!("Demonstrating Parameter Space Exploration");
    println!("------------------------------------------\n");

    // Create parameter space explorer
    let mut explorer = ParameterSpaceExplorer::new(42);
    let configs = explorer.generate_configurations();

    println!("Generated {} parameter configurations", configs.len());
    println!("\nSample configurations:");

    // Show first few configurations
    for (i, config) in configs.iter().take(3).enumerate() {
        println!("\nConfiguration {}:", i + 1);
        print_configuration(config);
    }

    // Generate matrices for a sample configuration
    println!("\n\nGenerating matrices for configuration 1...");
    let mut gen = MatrixGenerator::new(42);
    let matrices = gen.generate_matrices(&configs[0]);

    for (i, matrix) in matrices.iter().enumerate() {
        println!(
            "  Matrix {}: {}×{}, {} non-zeros (density: {:.6})",
            i + 1,
            matrix.n_rows,
            matrix.n_cols,
            matrix.nnz(),
            matrix.nnz() as f64 / (matrix.n_rows * matrix.n_cols) as f64
        );
    }
}

fn generate_test_suite() {
    println!("Generating Complete Test Suite");
    println!("------------------------------\n");

    let mut gen = TestSuiteGenerator::new(42);
    let suite = gen.generate_test_suite();

    println!("Generated {} test cases", suite.test_cases.len());
    println!("\nTest case distribution:");

    // Analyze the distribution
    let mut size_counts = std::collections::HashMap::new();
    let mut density_counts = std::collections::HashMap::new();

    for case in &suite.test_cases {
        *size_counts.entry(case.config.matrix.size).or_insert(0) += 1;
        
        let density_category = if case.config.matrix.density < ULTRA_SPARSE_DENSITY {
            "ultra-sparse"
        } else if case.config.matrix.density < SPARSE_DENSITY {
            "sparse"
        } else if case.config.matrix.density < MEDIUM_DENSITY {
            "medium"
        } else {
            "dense"
        };
        *density_counts.entry(density_category).or_insert(0) += 1;
    }

    println!("\nBy size:");
    for (size, count) in size_counts.iter() {
        println!("  {} matrices: {}", size, count);
    }

    println!("\nBy density:");
    for (density, count) in density_counts.iter() {
        println!("  {}: {}", density, count);
    }

    // Save a few test matrices
    println!("\nSaving sample matrices to disk...");
    std::fs::create_dir_all("test_matrices").ok();

    for (i, case) in suite.test_cases.iter().take(3).enumerate() {
        for (j, matrix) in case.matrices.iter().take(1).enumerate() {
            let path = format!("test_matrices/test_{}_{}.mtx", i, j);
            if let Err(e) = MatrixMarketIO::write_matrix(&path, matrix) {
                eprintln!("Failed to write matrix: {}", e);
            } else {
                println!("  Saved {}", path);
            }
        }
    }
}

fn run_benchmark() {
    println!("Running Parameter Space Benchmarks");
    println!("-----------------------------------\n");

    let mut explorer = ParameterSpaceExplorer::new(42);
    let configs = explorer.generate_configurations();

    // Select a subset of configurations for benchmarking
    let benchmark_configs: Vec<_> = configs
        .into_iter()
        .filter(|c| c.matrix.size <= MEDIUM_MATRIX_SIZE) // Only benchmark smaller matrices
        .take(10)
        .collect();

    println!("Benchmarking {} configurations\n", benchmark_configs.len());

    let mut gen = MatrixGenerator::new(42);

    for (i, config) in benchmark_configs.iter().enumerate() {
        println!("Configuration {}/{}:", i + 1, benchmark_configs.len());
        print_configuration(config);

        // Generate test matrices
        let matrices = gen.generate_matrices(config);

        // Benchmark SpGEMM
        for (j, matrix_a) in matrices.iter().take(1).enumerate() {
            // Create a compatible matrix B (transpose for valid multiplication)
            let matrix_b = create_compatible_matrix(matrix_a);

            let start = Instant::now();
            let _result = reference_spgemm(matrix_a, &matrix_b);
            let elapsed = start.elapsed();

            println!(
                "  Matrix {} SpGEMM: {:.3} ms",
                j,
                elapsed.as_secs_f64() * 1000.0
            );
        }
        println!();
    }
}

fn demonstrate_patterns() {
    println!("Demonstrating Pattern-Based Matrix Generation");
    println!("----------------------------------------------\n");

    let mut gen = PatternMatrixGenerator::new(42);
    let size = 1000;

    // Generate different patterns
    let patterns = vec![
        ("Banded (bandwidth=10)", gen.generate_banded(size, 10)),
        ("Block Diagonal (block=50)", gen.generate_block_diagonal(size, 50)),
        ("Power Law (α=1.5)", gen.generate_power_law(size, 1.5)),
        ("Power Law (α=2.5)", gen.generate_power_law(size, 2.5)),
    ];

    for (name, matrix) in patterns {
        println!("{}:", name);
        println!("  Size: {}×{}", matrix.n_rows, matrix.n_cols);
        println!("  Non-zeros: {}", matrix.nnz());
        println!("  Density: {:.6}", 
            matrix.nnz() as f64 / (matrix.n_rows * matrix.n_cols) as f64);
        
        // Analyze row distribution
        let mut min_nnz = usize::MAX;
        let mut max_nnz = 0;
        let mut total_nnz = 0;

        for i in 0..matrix.n_rows {
            let row_nnz = matrix.row_ptr[i + 1] - matrix.row_ptr[i];
            min_nnz = min_nnz.min(row_nnz);
            max_nnz = max_nnz.max(row_nnz);
            total_nnz += row_nnz;
        }

        println!("  Row NNZ: min={}, max={}, avg={:.1}", 
            min_nnz, max_nnz, total_nnz as f64 / matrix.n_rows as f64);
        println!();
    }
}

fn demonstrate_suitesparse_style() {
    println!("Generating SuiteSparse-Style Test Matrices");
    println!("-------------------------------------------\n");

    let mut gen = SuiteSparseStyleGenerator::new(42);
    let size = 5000;

    let collection = gen.generate_collection(size);

    for (domain, matrix) in collection {
        println!("{} matrix:", domain);
        println!("  Size: {}×{}", matrix.n_rows, matrix.n_cols);
        println!("  Non-zeros: {}", matrix.nnz());
        println!("  Density: {:.6}", 
            matrix.nnz() as f64 / (matrix.n_rows * matrix.n_cols) as f64);

        // Save to file
        let path = format!("test_matrices/{}_matrix.mtx", domain);
        std::fs::create_dir_all("test_matrices").ok();
        
        if let Err(e) = MatrixMarketIO::write_matrix(&path, &matrix) {
            eprintln!("  Failed to save: {}", e);
        } else {
            println!("  Saved to: {}", path);
        }
        println!();
    }

    // Show recommended SuiteSparse matrices
    println!("\nRecommended SuiteSparse Matrices for Testing:");
    println!("----------------------------------------------");
    
    use magnus::suitesparse_integration::SuiteSparseDownloader;
    
    for matrix in SuiteSparseDownloader::recommended_matrices() {
        println!("  {}/{}: {}×{}, {} nnz ({})",
            matrix.group,
            matrix.name,
            matrix.n_rows,
            matrix.n_cols,
            matrix.nnz,
            matrix.kind
        );
        println!("    {}", matrix.description);
    }
}

fn print_configuration(config: &ParameterConfiguration) {
    println!("  Matrix: size={}, density={:.4}, avg_nnz={}",
        config.matrix.size,
        config.matrix.density,
        config.matrix.avg_nnz_per_row
    );
    println!("  Algorithm: dense_threshold={}, gpu_threshold={}, simd_min={}",
        config.algorithm.dense_threshold,
        config.algorithm.gpu_threshold,
        config.algorithm.simd_min_elements
    );
    println!("  Memory: l2_cache={}, chunk_size={}",
        config.memory.l2_cache_size,
        config.memory.chunk_size
    );
}

fn create_compatible_matrix(matrix: &SparseMatrixCSR<f64>) -> SparseMatrixCSR<f64> {
    // Create a simple compatible matrix (transpose structure with random values)
    let n = matrix.n_cols;
    let m = matrix.n_rows;
    
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        // Each row has a few random non-zeros
        let nnz = rng.gen_range(1..5.min(m));
        let mut cols = std::collections::HashSet::new();
        
        while cols.len() < nnz {
            cols.insert(rng.gen_range(0..m));
        }

        let mut sorted_cols: Vec<_> = cols.into_iter().collect();
        sorted_cols.sort_unstable();

        for col in sorted_cols {
            col_idx.push(col);
            values.push(rng.gen_range(-1.0..1.0));
        }

        row_ptr.push(col_idx.len());
    }

    SparseMatrixCSR::new(n, m, row_ptr, col_idx, values)
}