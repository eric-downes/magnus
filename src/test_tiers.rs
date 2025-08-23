//! Unified test and benchmark tier system for MAGNUS
//!
//! Integrates dimensional analysis (Buckingham π) testing with traditional
//! matrix size-based testing into a coherent tier system.

use crate::dimensional_analysis::BuckinghamPiGroups;
use crate::matrix::SparseMatrixCSR;
use crate::reduced_parameter_space::{EfficientParameterExplorer, PiConfiguration, SmartPiSampler};
use std::time::Duration;

/// Test tier levels with time budgets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestTier {
    /// Quick validation - under 30 seconds
    /// Used for: TDD, pre-commit hooks
    Quick,

    /// Commit validation - under 10 minutes
    /// Used for: Before commits, CI on every push
    Commit,

    /// Pull request validation - under 30 minutes  
    /// Used for: PR checks, branch merges
    PullRequest,

    /// Release validation - under 2 hours
    /// Used for: Release candidates, major version bumps
    Release,
}

impl TestTier {
    /// Get the time budget for this tier
    pub fn time_budget(&self) -> Duration {
        match self {
            TestTier::Quick => Duration::from_secs(30),
            TestTier::Commit => Duration::from_secs(600), // 10 minutes
            TestTier::PullRequest => Duration::from_secs(1800), // 30 minutes
            TestTier::Release => Duration::from_secs(7200), // 2 hours
        }
    }

    /// Get tier from environment variable or default
    pub fn from_env() -> Self {
        match std::env::var("TEST_TIER").as_deref() {
            Ok("quick") => TestTier::Quick,
            Ok("commit") => TestTier::Commit,
            Ok("pr") | Ok("pull_request") => TestTier::PullRequest,
            Ok("release") => TestTier::Release,
            _ => TestTier::Quick, // Default to quick for safety
        }
    }
}

/// Configuration for each test tier
pub struct TierConfig {
    /// Traditional matrix sizes to test
    pub matrix_sizes: Vec<usize>,

    /// Maximum non-zeros for generated matrices
    pub max_nnz: usize,

    /// Buckingham π configurations to test
    pub pi_configs: Vec<BuckinghamPiGroups>,

    /// Number of random matrices per configuration
    pub matrices_per_config: usize,

    /// Whether to run parallel tests
    pub test_parallel: bool,

    /// Whether to run GPU tests
    pub test_gpu: bool,

    /// Whether to run memory stress tests
    pub test_memory_stress: bool,
}

impl TierConfig {
    /// Get configuration for a specific tier
    pub fn for_tier(tier: TestTier) -> Self {
        match tier {
            TestTier::Quick => Self::quick_config(),
            TestTier::Commit => Self::commit_config(),
            TestTier::PullRequest => Self::pr_config(),
            TestTier::Release => Self::release_config(),
        }
    }

    /// Quick tier: 5 critical π configs + small matrices
    fn quick_config() -> Self {
        let sampler = SmartPiSampler::new(42);
        let critical_configs = sampler.get_critical_configurations();

        Self {
            matrix_sizes: vec![500, 1000, 2000, 5000],
            max_nnz: 250_000,
            pi_configs: critical_configs, // 5 configurations
            matrices_per_config: 1,
            test_parallel: true,
            test_gpu: false,
            test_memory_stress: false,
        }
    }

    /// Commit tier: ~50 smart π configs + medium matrices
    fn commit_config() -> Self {
        let mut sampler = SmartPiSampler::new(42);
        let interesting = sampler.sample_interesting_regions();

        // Take a subset of interesting configurations
        let subset_size = 50.min(interesting.len());
        let pi_configs: Vec<_> = interesting
            .into_iter()
            .take(subset_size)
            .map(|config| config.pi_groups)
            .collect();

        Self {
            matrix_sizes: vec![1000, 5000, 10000, 15000],
            max_nnz: 2_000_000,
            pi_configs,
            matrices_per_config: 2,
            test_parallel: true,
            test_gpu: cfg!(all(target_arch = "aarch64", target_os = "macos")),
            test_memory_stress: false,
        }
    }

    /// PR tier: ~500 smart π configs + large matrices
    fn pr_config() -> Self {
        let mut sampler = SmartPiSampler::new(42);
        let interesting = sampler.sample_interesting_regions();

        // Take more configurations for PR testing
        let subset_size = 500.min(interesting.len());
        let pi_configs: Vec<_> = interesting
            .into_iter()
            .take(subset_size)
            .map(|config| config.pi_groups)
            .collect();

        Self {
            matrix_sizes: vec![5000, 10000, 25000, 50000],
            max_nnz: 5_000_000,
            pi_configs,
            matrices_per_config: 2,
            test_parallel: true,
            test_gpu: cfg!(all(target_arch = "aarch64", target_os = "macos")),
            test_memory_stress: true,
        }
    }

    /// Release tier: All 2.6k interesting π configs + stress tests
    fn release_config() -> Self {
        let mut sampler = SmartPiSampler::new(42);
        let interesting = sampler.sample_interesting_regions();

        let pi_configs: Vec<_> = interesting
            .into_iter()
            .map(|config| config.pi_groups)
            .collect();

        Self {
            matrix_sizes: vec![10000, 50000, 100000, 200000],
            max_nnz: 20_000_000,
            pi_configs, // All ~2,673 configurations
            matrices_per_config: 3,
            test_parallel: true,
            test_gpu: cfg!(all(target_arch = "aarch64", target_os = "macos")),
            test_memory_stress: true,
        }
    }

    /// Estimate total test time for this configuration
    pub fn estimate_time(&self, time_per_test: Duration) -> Duration {
        let pi_tests = self.pi_configs.len() * self.matrix_sizes.len() * self.matrices_per_config;

        let traditional_tests = self.matrix_sizes.len() * 5; // Estimate 5 patterns per size

        let total_tests = pi_tests + traditional_tests;
        time_per_test * total_tests as u32
    }

    /// Get a descriptive summary of this configuration
    pub fn summary(&self) -> String {
        format!(
            "Tier Configuration:\n\
             - Matrix sizes: {:?}\n\
             - Max NNZ: {}\n\
             - π configurations: {}\n\
             - Matrices per config: {}\n\
             - Test parallel: {}\n\
             - Test GPU: {}\n\
             - Memory stress: {}\n\
             - Estimated time: {:.1} minutes",
            self.matrix_sizes,
            self.max_nnz,
            self.pi_configs.len(),
            self.matrices_per_config,
            self.test_parallel,
            self.test_gpu,
            self.test_memory_stress,
            self.estimate_time(Duration::from_millis(500)).as_secs_f64() / 60.0
        )
    }
}

/// Benchmark tier configuration
pub struct BenchmarkTier {
    pub tier: TestTier,
    pub config: TierConfig,
}

impl BenchmarkTier {
    /// Get benchmark configuration from environment
    pub fn from_env() -> Self {
        // Check for BENCH_TIER first, then TEST_TIER
        let tier = match std::env::var("BENCH_TIER").as_deref() {
            Ok("quick") => TestTier::Quick,
            Ok("commit") | Ok("standard") => TestTier::Commit,
            Ok("pr") | Ok("large") => TestTier::PullRequest,
            Ok("release") | Ok("full") => TestTier::Release,
            _ => TestTier::from_env(),
        };

        Self {
            config: TierConfig::for_tier(tier),
            tier,
        }
    }

    /// Should we run this specific benchmark?
    pub fn should_run(&self, benchmark_name: &str) -> bool {
        match self.tier {
            TestTier::Quick => {
                // Only run quick benchmarks
                benchmark_name.contains("quick")
                    || benchmark_name.contains("small")
                    || benchmark_name.contains("critical")
            }
            TestTier::Commit => {
                // Skip stress and huge benchmarks
                !benchmark_name.contains("stress")
                    && !benchmark_name.contains("huge")
                    && !benchmark_name.contains("release")
            }
            TestTier::PullRequest => {
                // Skip only release-specific benchmarks
                !benchmark_name.contains("release")
            }
            TestTier::Release => {
                // Run everything
                true
            }
        }
    }
}

/// Generate test matrices for a given tier and π configuration
pub fn generate_test_matrices(
    tier: TestTier,
    pi_config: &BuckinghamPiGroups,
) -> Vec<SparseMatrixCSR<f64>> {
    let config = TierConfig::for_tier(tier);
    let mut explorer = EfficientParameterExplorer::new(42);
    let mut all_matrices = Vec::new();

    for &size in &config.matrix_sizes {
        let pi_configuration = PiConfiguration {
            pi_groups: pi_config.clone(),
            base_sizes: vec![size],
        };

        let matrices = explorer.generate_matrices_for_pi(&pi_configuration, size);
        all_matrices.extend(matrices.into_iter().take(config.matrices_per_config));
    }

    all_matrices
}

/// Test result tracking
pub struct TierTestResult {
    pub tier: TestTier,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub duration: Duration,
}

impl TierTestResult {
    pub fn new(tier: TestTier) -> Self {
        Self {
            tier,
            passed: 0,
            failed: 0,
            skipped: 0,
            duration: Duration::ZERO,
        }
    }

    pub fn summary(&self) -> String {
        let total = self.passed + self.failed + self.skipped;
        format!(
            "{:?} Tier Results: {} passed, {} failed, {} skipped ({} total) in {:.1}s",
            self.tier,
            self.passed,
            self.failed,
            self.skipped,
            total,
            self.duration.as_secs_f64()
        )
    }

    pub fn is_success(&self) -> bool {
        self.failed == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_configs() {
        let quick = TierConfig::for_tier(TestTier::Quick);
        assert_eq!(quick.pi_configs.len(), 5);
        assert!(quick.max_nnz <= 250_000);

        let commit = TierConfig::for_tier(TestTier::Commit);
        assert!(commit.pi_configs.len() <= 50);
        assert!(commit.max_nnz <= 2_000_000);

        println!("Quick tier: {}", quick.summary());
        println!("Commit tier: {}", commit.summary());
    }

    #[test]
    fn test_time_estimates() {
        for tier in [
            TestTier::Quick,
            TestTier::Commit,
            TestTier::PullRequest,
            TestTier::Release,
        ] {
            let config = TierConfig::for_tier(tier);
            let estimate = config.estimate_time(Duration::from_millis(500));
            let budget = tier.time_budget();

            println!(
                "{:?} tier: estimated {:.1}min, budget {:.1}min",
                tier,
                estimate.as_secs_f64() / 60.0,
                budget.as_secs_f64() / 60.0
            );
        }
    }
}
