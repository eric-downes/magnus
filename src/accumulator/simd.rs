//! SIMD acceleration for sorting and accumulation operations

use num_traits::Num;
use std::ops::AddAssign;

/// Trait for SIMD-accelerated sorting and accumulation
pub trait SimdAccelerator<T>
where
    T: Copy + Num + AddAssign,
{
    /// Sort column indices and values, then accumulate duplicates
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[T]) -> (Vec<usize>, Vec<T>);
}

/// Generic fallback implementation
pub struct FallbackAccumulator;

impl FallbackAccumulator {
    pub fn new() -> Self {
        FallbackAccumulator
    }
}

impl<T> SimdAccelerator<T> for FallbackAccumulator
where
    T: Copy + Num + AddAssign,
{
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[T]) -> (Vec<usize>, Vec<T>) {
        if col_indices.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Create pairs and sort by column index
        let mut pairs: Vec<(usize, T)> = col_indices
            .iter()
            .zip(values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();
        
        pairs.sort_by_key(|&(idx, _)| idx);

        // Accumulate duplicates
        let mut result_indices = Vec::new();
        let mut result_values = Vec::new();

        let mut current_idx = pairs[0].0;
        let mut current_val = pairs[0].1;

        for i in 1..pairs.len() {
            if pairs[i].0 == current_idx {
                current_val += pairs[i].1;
            } else {
                result_indices.push(current_idx);
                result_values.push(current_val);
                current_idx = pairs[i].0;
                current_val = pairs[i].1;
            }
        }

        // Don't forget the last group
        result_indices.push(current_idx);
        result_values.push(current_val);

        (result_indices, result_values)
    }
}

/// ARM NEON accelerated implementation (re-export from neon module)
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use super::neon::NeonAccumulator;


/// AVX-512 accelerated implementation (placeholder)
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub struct Avx512Accumulator;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl Avx512Accumulator {
    pub fn new() -> Self {
        Avx512Accumulator
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl<T> SimdAccelerator<T> for Avx512Accumulator
where
    T: Copy + Num + AddAssign,
{
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[T]) -> (Vec<usize>, Vec<T>) {
        // Placeholder for AVX-512 implementation
        // Would use Intel's x86-simd-sort library
        FallbackAccumulator.sort_and_accumulate(col_indices, values)
    }
}

/// Create an appropriate SIMD accelerator for f32
pub fn create_simd_accelerator_f32() -> Box<dyn SimdAccelerator<f32>> {
    use crate::matrix::config::detect_architecture;
    use crate::matrix::config::Architecture;
    
    match detect_architecture() {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        Architecture::ArmNeon => {
            // Use environment variable to optionally select Accelerate framework
            if std::env::var("MAGNUS_USE_ACCELERATE").is_ok() {
                Box::new(super::accelerate::AccelerateAccumulator::new())
            } else {
                Box::new(super::neon::NeonAccumulator::new())
            }
        },
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        Architecture::X86WithAVX512 => Box::new(Avx512Accumulator::new()),
        _ => Box::new(FallbackAccumulator::new()),
    }
}

/// Create an appropriate SIMD accelerator for the current architecture
pub fn create_simd_accelerator<T>() -> Box<dyn SimdAccelerator<T>>
where
    T: Copy + Num + AddAssign + 'static,
{
    // For now, only use fallback for non-f32 types
    Box::new(FallbackAccumulator::new())
}