//! Dimensional analysis exploration tool
//!
//! This tool demonstrates the power of dimensional analysis in reducing
//! the parameter space from ~20k to ~1.5k critical configurations.

use magnus::dimensional_analysis::{
    BuckinghamPiGroups, DimensionalConstants, DimensionalReduction, ImplicitConversions,
};
use magnus::matrix::SparseMatrixCSR;
use magnus::reduced_parameter_space::{
    EfficientParameterExplorer, PiConfiguration, ReductionSummary, ScalingAnalyzer, SmartPiSampler,
};
use magnus::sparse_gemm_parallel;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("demo");

    match mode {
        "demo" => run_demo(),
        "analyze" => analyze_units(),
        "explore" => explore_pi_space(),
        "benchmark" => benchmark_critical(),
        "scaling" => test_scaling_laws(),
        "full" => run_full_exploration(),
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("Dimensional Analysis Explorer for MAGNUS");
    println!("=========================================");
    println!();
    println!("Usage: explore_dimensions [mode]");
    println!();
    println!("Modes:");
    println!("  demo      - Demonstrate dimensional reduction (default)");
    println!("  analyze   - Analyze physical units of constants");
    println!("  explore   - Explore π-space configurations");
    println!("  benchmark - Benchmark critical configurations");
    println!("  scaling   - Test dimensional scaling laws");
    println!("  full      - Run complete exploration (warning: slow)");
}

fn run_demo() {
    println!("MAGNUS Dimensional Analysis Demonstration");
    println!("==========================================");
    println!();

    // Show reduction summary
    println!("{}", ReductionSummary::generate_summary());
    println!();

    // Show dimensional analysis
    println!("{}", DimensionalReduction::analyze_reduction());
    println!();

    // Show critical configurations
    println!("Critical π-Configurations for Testing");
    println!("--------------------------------------");
    let sampler = SmartPiSampler::new(42);
    let critical = sampler.get_critical_configurations();

    for (i, config) in critical.iter().enumerate() {
        println!("\nConfiguration {}:", i + 1);
        println!("  π₁ (cache):      {:.3}", config.pi1_cache_utilization);
        println!("  π₂ (SIMD):       {:.3}", config.pi2_simd_efficiency);
        println!("  π₃ (memory):     {:.3e}", config.pi3_memory_hierarchy);
        println!("  π₄ (density):    {:.3e}", config.pi4_density_threshold);
        println!("  π₅ (GPU):        {:.3e}", config.pi5_gpu_utilization);
        println!("  π₆ (prefetch):   {:.3}", config.pi6_prefetch_efficiency);
        println!("  π₇ (accumulator): {:.1}", config.pi7_accumulator_ratio);
        println!("  π₈ (NNZ dist):   {:.3}", config.pi8_nnz_distribution);
    }
}

fn analyze_units() {
    println!("Physical Unit Analysis of MAGNUS Constants");
    println!("===========================================");
    println!();

    let units = DimensionalConstants::categorize_units();

    // Group by unit type
    let mut bytes_constants = Vec::new();
    let mut element_constants = Vec::new();
    let mut pure_constants = Vec::new();

    for (name, unit, value) in units {
        match unit {
            magnus::dimensional_analysis::PhysicalUnit::Bytes => {
                bytes_constants.push((name, value));
            }
            magnus::dimensional_analysis::PhysicalUnit::Elements => {
                element_constants.push((name, value));
            }
            magnus::dimensional_analysis::PhysicalUnit::Pure => {
                pure_constants.push((name, value));
            }
            _ => {}
        }
    }

    println!("Constants with dimension [Bytes]:");
    println!("---------------------------------");
    for (name, value) in &bytes_constants {
        println!("  {:<40} = {}", name, value);
    }

    println!("\nConstants with dimension [Elements]:");
    println!("-------------------------------------");
    for (name, value) in &element_constants {
        println!("  {:<40} = {}", name, value);
    }

    println!("\nDimensionless constants [Pure]:");
    println!("--------------------------------");
    for (name, value) in &pure_constants {
        println!("  {:<40} = {}", name, value);
    }

    println!("\nImplicit Unit Conversions:");
    println!("---------------------------");
    let conversions = ImplicitConversions::identify_conversions();
    for (conversion, description) in conversions {
        println!("  • {}", conversion);
        println!("    → {}", description);
    }
}

fn explore_pi_space() {
    println!("Exploring π-Space Configurations");
    println!("=================================");
    println!();

    let explorer = EfficientParameterExplorer::new(42);
    let configs = explorer.generate_efficient_configurations();

    println!("Total π-configurations: {}", configs.len());
    println!("Base sizes per config: {:?}", configs[0].base_sizes);
    println!(
        "Total test matrices: {}",
        configs.len()
            * configs[0].base_sizes.len()
            * magnus::reduced_parameter_space::NUM_MATRICES_PER_PI
    );
    println!();

    // Sample a few configurations
    println!("Sample π-configurations:");
    for i in [0, 100, 500, 1000, 5000].iter() {
        if *i < configs.len() {
            let config = &configs[*i];
            println!(
                "\nConfig {}: π = [{:.2}, {:.2}, {:.2e}, {:.2e}, {:.2e}, {:.2}, {:.0}, {:.2}]",
                i,
                config.pi_groups.pi1_cache_utilization,
                config.pi_groups.pi2_simd_efficiency,
                config.pi_groups.pi3_memory_hierarchy,
                config.pi_groups.pi4_density_threshold,
                config.pi_groups.pi5_gpu_utilization,
                config.pi_groups.pi6_prefetch_efficiency,
                config.pi_groups.pi7_accumulator_ratio,
                config.pi_groups.pi8_nnz_distribution
            );
        }
    }

    // Smart sampling
    println!("\n\nSmart Sampling Results:");
    println!("-----------------------");
    let mut sampler = SmartPiSampler::new(42);
    let interesting = sampler.sample_interesting_regions();

    println!("Interesting configurations: {}", interesting.len());
    println!(
        "Reduction from full space: {:.1}x",
        configs.len() as f64 / interesting.len() as f64
    );
}

fn benchmark_critical() {
    println!("Benchmarking Critical π-Configurations");
    println!("=======================================");
    println!();

    let mut explorer = EfficientParameterExplorer::new(42);
    let sampler = SmartPiSampler::new(42);
    let critical = sampler.get_critical_configurations();

    println!("Testing {} critical configurations...", critical.len());
    println!();

    for (i, pi_groups) in critical.iter().enumerate() {
        println!("Configuration {}: Testing scaling behavior", i + 1);

        let config = PiConfiguration {
            pi_groups: pi_groups.clone(),
            base_sizes: vec![1000, 5000, 10000],
        };

        let mut times = Vec::new();

        for &size in &config.base_sizes {
            let matrices = explorer.generate_matrices_for_pi(&config, size);

            if matrices.is_empty() {
                continue;
            }

            // Benchmark first matrix
            let a = &matrices[0];
            let b = create_compatible_matrix(a);

            let start = Instant::now();
            let _c = sparse_gemm_parallel(a, &b);
            let elapsed = start.elapsed();

            times.push(elapsed.as_secs_f64());

            println!(
                "  n={:5}: {:.3}ms ({} nnz)",
                size,
                elapsed.as_millis(),
                a.nnz()
            );
        }

        // Analyze scaling
        if times.len() == 3 {
            let report = ScalingAnalyzer::verify_scaling(pi_groups, times[0], times[1], times[2]);

            println!("  Scaling analysis:");
            println!("    Expected 5k/1k:  {:.2}x", report.expected_scale_5k);
            println!("    Actual 5k/1k:    {:.2}x", report.actual_scale_5k);
            println!("    Expected 10k/1k: {:.2}x", report.expected_scale_10k);
            println!("    Actual 10k/1k:   {:.2}x", report.actual_scale_10k);
            println!("    Scaling valid:   {}", report.scaling_valid);
        }
        println!();
    }
}

fn test_scaling_laws() {
    println!("Testing Dimensional Scaling Laws");
    println!("=================================");
    println!();

    // Create a test π-configuration
    let pi_groups = BuckinghamPiGroups {
        pi1_cache_utilization: 0.5,
        pi2_simd_efficiency: 0.5,
        pi3_memory_hierarchy: 1e-4,
        pi4_density_threshold: 1e-3,
        pi5_gpu_utilization: 1e4,
        pi6_prefetch_efficiency: 1.4,
        pi7_accumulator_ratio: 100.0,
        pi8_nnz_distribution: 1.0, // O(sqrt(n)) scaling
    };

    println!("Test π-configuration:");
    println!(
        "  π₁ = {:.2} (cache utilization)",
        pi_groups.pi1_cache_utilization
    );
    println!(
        "  π₈ = {:.2} (NNZ ~ √n scaling)",
        pi_groups.pi8_nnz_distribution
    );
    println!();

    // Generate matrices at different scales
    let mut explorer = EfficientParameterExplorer::new(42);
    let sizes = vec![500, 1000, 2000, 4000, 8000];
    let mut times = Vec::new();

    println!("Matrix Size | Time (ms) | NNZ     | Time/n^1.5");
    println!("------------|-----------|---------|------------");

    for &size in &sizes {
        let config = PiConfiguration {
            pi_groups: pi_groups.clone(),
            base_sizes: vec![size],
        };

        let matrices = explorer.generate_matrices_for_pi(&config, size);
        if matrices.is_empty() {
            continue;
        }

        let a = &matrices[0];
        let b = create_compatible_matrix(a);

        let start = Instant::now();
        let _c = sparse_gemm_parallel(a, &b);
        let elapsed = start.elapsed();

        let time_ms = elapsed.as_secs_f64() * 1000.0;
        let normalized = time_ms / (size as f64).powf(1.5);

        times.push(time_ms);

        println!(
            "{:11} | {:9.2} | {:7} | {:.6}",
            size,
            time_ms,
            a.nnz(),
            normalized
        );
    }

    // Check if scaling follows expected pattern
    if times.len() >= 2 {
        println!("\nScaling Analysis:");
        for i in 1..times.len() {
            let size_ratio = sizes[i] as f64 / sizes[i - 1] as f64;
            let time_ratio = times[i] / times[i - 1];
            let expected_ratio = size_ratio.powf(1.5); // O(n^1.5) for π₈=1
            let error = ((time_ratio - expected_ratio) / expected_ratio * 100.0).abs();

            println!(
                "  {}→{}: actual={:.2}x, expected={:.2}x, error={:.1}%",
                sizes[i - 1],
                sizes[i],
                time_ratio,
                expected_ratio,
                error
            );
        }
    }
}

fn run_full_exploration() {
    println!("Full π-Space Exploration (Warning: This takes time!)");
    println!("=====================================================");
    println!();

    let mut explorer = EfficientParameterExplorer::new(42);
    let configs = explorer.generate_efficient_configurations();

    println!("Configurations to test: {}", configs.len());
    println!("Estimated time: {:.1} minutes", configs.len() as f64 / 60.0);
    println!();

    let start_time = Instant::now();
    let mut tested = 0;
    let mut failures = 0;

    // Test subset to avoid extremely long runtime
    let test_subset = 100; // Test first 100 configurations

    for (i, config) in configs.iter().take(test_subset).enumerate() {
        // Test smallest size only for speed
        let size = config.base_sizes[0];
        let matrices = explorer.generate_matrices_for_pi(config, size);

        if matrices.is_empty() {
            failures += 1;
            continue;
        }

        let a = &matrices[0];
        let b = create_compatible_matrix(a);

        let test_start = Instant::now();
        let c = sparse_gemm_parallel(a, &b);
        let _test_time = test_start.elapsed();

        tested += 1;

        if i % 10 == 0 {
            let progress = (i + 1) as f64 / test_subset as f64 * 100.0;
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = tested as f64 / elapsed;

            println!(
                "Progress: {:.1}% | Tested: {} | Rate: {:.1} tests/s | Result NNZ: {}",
                progress,
                tested,
                rate,
                c.nnz()
            );
        }
    }

    let total_time = start_time.elapsed();
    println!("\nExploration Complete!");
    println!("---------------------");
    println!("Configurations tested: {}", tested);
    println!("Failed generations: {}", failures);
    println!("Total time: {:.1}s", total_time.as_secs_f64());
    println!(
        "Average per test: {:.1}ms",
        total_time.as_millis() as f64 / tested as f64
    );
}

/// Create a compatible matrix for multiplication
fn create_compatible_matrix(a: &SparseMatrixCSR<f64>) -> SparseMatrixCSR<f64> {
    // Create a simple diagonal matrix for testing
    let n = a.n_cols;
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        col_idx.push(i);
        values.push(1.0);
        row_ptr.push(i + 1);
    }

    SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
}
