//! Configuration and system parameters for MAGNUS

/// The target architecture for performance optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Architecture {
    /// Intel/AMD x86_64 with AVX-512 support
    X86WithAVX512,
    /// Intel/AMD x86_64 without AVX-512 support
    X86WithoutAVX512,
    /// ARM architecture with NEON support (e.g., Apple Silicon)
    ArmNeon,
    /// Generic implementation for any architecture
    Generic,
}

/// System parameters for performance tuning
#[derive(Debug, Clone)]
pub struct SystemParameters {
    /// Size of cache line in bytes
    pub cache_line_size: usize,
    /// Size of L2 cache in bytes
    pub l2_cache_size: usize,
    /// Number of threads to use
    pub n_threads: usize,
}

impl Default for SystemParameters {
    fn default() -> Self {
        Self {
            cache_line_size: 64,    // Common cache line size
            l2_cache_size: 256_000, // 256KB L2 cache (conservative default)
            n_threads: num_cpus::get(), // Use all available cores
        }
    }
}

/// Method for sorting and accumulating intermediate products
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SortMethod {
    /// Standard sort followed by scan and reduce
    SortThenReduce,
    /// Modified compare-exchange operations with integrated accumulation
    ModifiedCompareExchange,
}

/// Categorization of rows based on computational characteristics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowCategory {
    /// Small intermediate products, use sort-based approach
    Sort,
    /// Fits in L2 cache, use dense accumulation
    DenseAccumulation,
    /// Requires fine-level reordering
    FineLevel,
    /// Requires both coarse and fine-level reordering
    CoarseLevel,
}

/// Configuration for the MAGNUS algorithm
#[derive(Debug, Clone)]
pub struct MagnusConfig {
    /// System parameters for performance tuning
    pub system_params: SystemParameters,
    
    /// Threshold to switch from sort to dense accumulation
    pub dense_accum_threshold: usize,
    
    /// Method for sorting and accumulating
    pub sort_method: SortMethod,
    
    /// Whether to enable coarse-level reordering
    pub enable_coarse_level: bool,
    
    /// Batch size for coarse-level reordering
    /// If None, a heuristic based on available memory and cache size is used
    pub coarse_batch_size: Option<usize>,
    
    /// Target architecture for optimization
    pub architecture: Architecture,
}

impl Default for MagnusConfig {
    fn default() -> Self {
        Self {
            system_params: SystemParameters::default(),
            dense_accum_threshold: 256, // Default from paper
            sort_method: SortMethod::SortThenReduce,
            enable_coarse_level: true,
            coarse_batch_size: None,    // Use heuristic by default
            architecture: detect_architecture(),
        }
    }
}

/// Detects the current CPU architecture
fn detect_architecture() -> Architecture {
    // This is a placeholder. In a real implementation, we would use
    // runtime CPU feature detection to determine the architecture.
    #[cfg(target_arch = "x86_64")]
    {
        // Check for AVX-512 support
        // In a real implementation, we'd use CPUID or similar
        Architecture::X86WithoutAVX512
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        Architecture::ArmNeon
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Architecture::Generic
    }
}