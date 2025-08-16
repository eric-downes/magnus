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

impl Architecture {
    /// Check if this architecture has SIMD support
    pub fn has_simd_support(&self) -> bool {
        !matches!(self, Architecture::Generic)
    }

    /// Check if this architecture has AVX-512 support
    pub fn has_avx512(&self) -> bool {
        matches!(self, Architecture::X86WithAVX512)
    }

    /// Check if this architecture has ARM NEON support
    pub fn has_neon(&self) -> bool {
        matches!(self, Architecture::ArmNeon)
    }

    /// Get the vector width in bytes for this architecture
    pub fn vector_width_bytes(&self) -> usize {
        match self {
            Architecture::X86WithAVX512 => 64,    // 512 bits
            Architecture::X86WithoutAVX512 => 32, // 256 bits (AVX2)
            Architecture::ArmNeon => 16,          // 128 bits
            Architecture::Generic => 8,           // Scalar
        }
    }

    /// Get optimal chunk size for this architecture
    pub fn optimal_chunk_size(&self) -> usize {
        match self {
            Architecture::X86WithAVX512 => 2048,
            Architecture::X86WithoutAVX512 => 1024,
            Architecture::ArmNeon => 512, // Tuned for Apple Silicon cache
            Architecture::Generic => 256,
        }
    }

    /// Get the threshold for switching to dense accumulator
    pub fn dense_accumulator_threshold(&self) -> usize {
        match self {
            Architecture::X86WithAVX512 => 256, // From paper
            Architecture::X86WithoutAVX512 => 256,
            Architecture::ArmNeon => 192, // Tuned for Apple Silicon
            Architecture::Generic => 256,
        }
    }
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
            cache_line_size: 64,        // Common cache line size
            l2_cache_size: 256_000,     // 256KB L2 cache (conservative default)
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
        let arch = detect_architecture();
        Self {
            system_params: SystemParameters::default(),
            dense_accum_threshold: arch.dense_accumulator_threshold(),
            sort_method: SortMethod::SortThenReduce,
            enable_coarse_level: true,
            coarse_batch_size: None, // Use heuristic by default
            architecture: arch,
        }
    }
}

impl MagnusConfig {
    /// Create a config optimized for a specific architecture
    pub fn for_architecture(arch: Architecture) -> Self {
        Self {
            system_params: SystemParameters::default(),
            dense_accum_threshold: arch.dense_accumulator_threshold(),
            sort_method: SortMethod::SortThenReduce,
            enable_coarse_level: true,
            coarse_batch_size: None,
            architecture: arch,
        }
    }
}

/// Detects the current CPU architecture
pub fn detect_architecture() -> Architecture {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        // Apple Silicon always has NEON
        return Architecture::ArmNeon;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Check for AVX-512 support using is_x86_feature_detected macro
        #[cfg(target_feature = "avx512f")]
        {
            return Architecture::X86WithAVX512;
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            // Runtime detection for x86
            if std::is_x86_feature_detected!("avx512f") {
                return Architecture::X86WithAVX512;
            } else {
                return Architecture::X86WithoutAVX512;
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        // Other ARM platforms with NEON
        return Architecture::ArmNeon;
    }

    // Fallback for other architectures
    #[allow(unreachable_code)]
    Architecture::Generic
}
