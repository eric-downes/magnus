//! Safe wrapper for NEON operations
//!
//! This module provides a safe interface to NEON SIMD operations,
//! handling all safety checks and fallbacks transparently.

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use super::{neon::NeonAccumulator, FallbackAccumulator, SimdAccelerator};

/// Safe NEON accelerator with runtime checks and automatic fallback
pub struct SafeNeonAccumulator {
    /// Use NEON if available, otherwise fallback
    backend: Box<dyn SimdAccelerator<f32>>,
}

impl SafeNeonAccumulator {
    /// Create a new safe NEON accelerator
    ///
    /// This will automatically detect NEON availability at runtime
    /// and fall back to scalar implementation if needed.
    pub fn new() -> Self {
        let backend: Box<dyn SimdAccelerator<f32>> = if Self::is_neon_available() {
            Box::new(NeonAccumulator::new())
        } else {
            // NEON not available, using fallback implementation
            Box::new(FallbackAccumulator::new())
        };

        SafeNeonAccumulator { backend }
    }

    /// Check if NEON is available at runtime
    #[inline]
    fn is_neon_available() -> bool {
        // On macOS ARM64, NEON is always available
        // But we add this for future extensibility
        #[cfg(target_os = "macos")]
        {
            // All Apple Silicon has NEON
            true
        }
        #[cfg(not(target_os = "macos"))]
        {
            // For other ARM64 platforms, check at runtime
            std::arch::is_aarch64_feature_detected!("neon")
        }
    }

    /// Validate input data before processing
    #[inline]
    fn validate_inputs(col_indices: &[usize], values: &[f32]) -> Result<(), &'static str> {
        if col_indices.len() != values.len() {
            return Err("Mismatched array lengths");
        }

        // Check for potential overflow when converting to u32
        for &idx in col_indices {
            if idx > u32::MAX as usize {
                return Err("Column index too large for NEON operations");
            }
        }

        // Check for NaN or infinite values that could cause issues
        for &val in values {
            if !val.is_finite() {
                return Err("Non-finite values detected");
            }
        }

        Ok(())
    }
}

impl SimdAccelerator<f32> for SafeNeonAccumulator {
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
        // Validate inputs first
        if let Err(e) = Self::validate_inputs(col_indices, values) {
            eprintln!("Warning: Input validation failed: {}, using fallback", e);
            return FallbackAccumulator::new().sort_and_accumulate(col_indices, values);
        }

        // Check size thresholds
        if col_indices.len() < 4 {
            // Too small for NEON to be beneficial
            return FallbackAccumulator::new().sort_and_accumulate(col_indices, values);
        }

        // Use the selected backend
        self.backend.sort_and_accumulate(col_indices, values)
    }
}

impl Default for SafeNeonAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_neon_empty_input() {
        let acc = SafeNeonAccumulator::new();
        let (indices, values) = acc.sort_and_accumulate(&[], &[]);
        assert!(indices.is_empty());
        assert!(values.is_empty());
    }

    #[test]
    fn test_safe_neon_small_input() {
        let acc = SafeNeonAccumulator::new();
        let col_indices = vec![2, 1];
        let values = vec![2.0, 1.0];

        let (result_idx, result_val) = acc.sort_and_accumulate(&col_indices, &values);
        assert_eq!(result_idx, vec![1, 2]);
        assert_eq!(result_val, vec![1.0, 2.0]);
    }

    #[test]
    fn test_safe_neon_validation() {
        let acc = SafeNeonAccumulator::new();

        // Test with large indices that would overflow u32
        let col_indices = vec![u32::MAX as usize + 1, u32::MAX as usize + 2];
        let values = vec![1.0, 2.0];

        // Should use fallback for large indices
        let (result_idx, result_val) = acc.sort_and_accumulate(&col_indices, &values);

        // Verify it still works correctly with fallback
        assert_eq!(result_idx.len(), 2);
        assert_eq!(result_val.len(), 2);
    }
}
