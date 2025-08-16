use magnus::matrix::config::{detect_architecture, Architecture};

#[test]
fn test_architecture_detection() {
    let arch = detect_architecture();

    // On Apple Silicon, we should detect ARM NEON
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        assert_eq!(arch, Architecture::ArmNeon);
    }

    // On x86_64, we should detect either with or without AVX-512
    #[cfg(target_arch = "x86_64")]
    {
        assert!(arch == Architecture::X86WithAVX512 || arch == Architecture::X86WithoutAVX512);
    }

    // On other platforms, we should get Generic
    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        target_arch = "x86_64"
    )))]
    {
        assert_eq!(arch, Architecture::Generic);
    }
}

#[test]
fn test_architecture_capabilities() {
    let arch = detect_architecture();

    // Test that each architecture reports correct capabilities
    match arch {
        Architecture::ArmNeon => {
            assert!(arch.has_simd_support());
            assert!(!arch.has_avx512());
            assert!(arch.has_neon());
        }
        Architecture::X86WithAVX512 => {
            assert!(arch.has_simd_support());
            assert!(arch.has_avx512());
            assert!(!arch.has_neon());
        }
        Architecture::X86WithoutAVX512 => {
            assert!(arch.has_simd_support());
            assert!(!arch.has_avx512());
            assert!(!arch.has_neon());
        }
        Architecture::Generic => {
            assert!(!arch.has_simd_support());
            assert!(!arch.has_avx512());
            assert!(!arch.has_neon());
        }
    }
}

#[test]
fn test_architecture_vector_width() {
    let arch = detect_architecture();

    // Test expected vector widths for different architectures
    match arch {
        Architecture::ArmNeon => {
            // NEON has 128-bit vectors (16 bytes)
            assert_eq!(arch.vector_width_bytes(), 16);
        }
        Architecture::X86WithAVX512 => {
            // AVX-512 has 512-bit vectors (64 bytes)
            assert_eq!(arch.vector_width_bytes(), 64);
        }
        Architecture::X86WithoutAVX512 => {
            // AVX2 has 256-bit vectors (32 bytes)
            assert_eq!(arch.vector_width_bytes(), 32);
        }
        Architecture::Generic => {
            // No SIMD, report scalar width
            assert_eq!(arch.vector_width_bytes(), 8);
        }
    }
}

#[test]
fn test_architecture_optimal_chunk_size() {
    let arch = detect_architecture();

    // Each architecture should provide reasonable chunk sizes
    let chunk_size = arch.optimal_chunk_size();

    match arch {
        Architecture::ArmNeon => {
            // For ARM NEON, expect chunk sizes that fit well in cache
            assert!(chunk_size >= 256);
            assert!(chunk_size <= 4096);
            // Should be power of 2 for efficient masking
            assert!(chunk_size.is_power_of_two());
        }
        Architecture::X86WithAVX512 => {
            assert!(chunk_size >= 512);
            assert!(chunk_size <= 8192);
            assert!(chunk_size.is_power_of_two());
        }
        _ => {
            assert!(chunk_size >= 128);
            assert!(chunk_size <= 2048);
        }
    }
}

#[test]
fn test_architecture_accumulator_threshold() {
    let arch = detect_architecture();

    // Different architectures should have different thresholds
    // for switching between sort and dense accumulators
    let threshold = arch.dense_accumulator_threshold();

    match arch {
        Architecture::ArmNeon => {
            // ARM typically has different cache characteristics
            assert!(threshold >= 128);
            assert!(threshold <= 512);
        }
        Architecture::X86WithAVX512 => {
            // Paper suggests 256 for x86 with AVX-512
            assert_eq!(threshold, 256);
        }
        _ => {
            assert!(threshold >= 128);
            assert!(threshold <= 1024);
        }
    }
}
