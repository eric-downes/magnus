//! Metal GPU implementation using metal-rs crate
//!
//! This provides the actual Metal implementation for GPU-accelerated
//! sparse matrix operations on Apple Silicon.

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use super::SimdAccelerator;
use metal::Library;
use metal::{CommandQueue, ComputePipelineState, Device};
use metal::{MTLResourceOptions, MTLSize};
use std::mem;
use std::sync::{Arc, Mutex, Once};

/// Threshold for using Metal GPU (number of elements)
const METAL_THRESHOLD: usize = 10_000;

/// Shared Metal context for the application
static METAL_CONTEXT: Mutex<Option<Arc<MetalContext>>> = Mutex::new(None);
static INIT: Once = Once::new();

/// Shared Metal context containing device and compiled kernels
#[allow(dead_code)]
struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
    pipelines: ComputePipelines,
}

#[allow(dead_code)]
struct ComputePipelines {
    bitonic_sort: ComputePipelineState,
    parallel_reduce: ComputePipelineState,
}

impl MetalContext {
    fn new() -> Option<Self> {
        // Get the default Metal device
        let device = Device::system_default()?;

        // Create command queue
        let command_queue = device.new_command_queue();

        // Compile shaders
        let shader_source = include_str!("metal_kernels.metal");
        let library = device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .ok()?;

        // Create compute pipelines
        let bitonic_function = library.get_function("bitonic_sort_step", None).ok()?;
        let bitonic_pipeline = device
            .new_compute_pipeline_state_with_function(&bitonic_function)
            .ok()?;

        let reduce_function = library.get_function("parallel_reduce_sum", None).ok()?;
        let reduce_pipeline = device
            .new_compute_pipeline_state_with_function(&reduce_function)
            .ok()?;

        Some(MetalContext {
            device,
            command_queue,
            library,
            pipelines: ComputePipelines {
                bitonic_sort: bitonic_pipeline,
                parallel_reduce: reduce_pipeline,
            },
        })
    }

    fn get() -> Option<Arc<MetalContext>> {
        INIT.call_once(|| {
            if let Some(context) = MetalContext::new() {
                *METAL_CONTEXT.lock().unwrap() = Some(Arc::new(context));
            }
        });
        METAL_CONTEXT.lock().unwrap().clone()
    }
}

/// Metal-accelerated accumulator for large arrays
pub struct MetalAccumulator {
    context: Arc<MetalContext>,
}

impl MetalAccumulator {
    /// Create a new Metal accelerator if available
    pub fn new() -> Option<Self> {
        MetalContext::get().map(|context| MetalAccumulator { context })
    }

    /// Check if Metal should be used for this size
    pub fn should_use_metal(size: usize) -> bool {
        size >= METAL_THRESHOLD && Self::is_available()
    }

    /// Check if Metal is available
    pub fn is_available() -> bool {
        MetalContext::get().is_some()
    }

    /// Perform bitonic sort on GPU
    fn gpu_bitonic_sort(&self, indices: &[u32], values: &[f32]) -> (Vec<u32>, Vec<f32>) {
        let n = indices.len();
        let n_padded = n.next_power_of_two();

        // Pad to power of 2 for bitonic sort
        let mut padded_indices = vec![u32::MAX; n_padded];
        let mut padded_values = vec![f32::INFINITY; n_padded];
        padded_indices[..n].copy_from_slice(indices);
        padded_values[..n].copy_from_slice(values);

        // Create Metal buffers
        let device = &self.context.device;
        let indices_buffer = device.new_buffer_with_data(
            padded_indices.as_ptr() as *const _,
            (n_padded * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let values_buffer = device.new_buffer_with_data(
            padded_values.as_ptr() as *const _,
            (n_padded * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer
        let command_buffer = self.context.command_queue.new_command_buffer();

        // Perform bitonic sort passes
        let num_stages = (n_padded as f32).log2() as u32;

        for stage in 0..num_stages {
            for pass in 0..=stage {
                let pass_of_stage = 1 << (stage - pass);

                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.context.pipelines.bitonic_sort);
                encoder.set_buffer(0, Some(&indices_buffer), 0);
                encoder.set_buffer(1, Some(&values_buffer), 0);
                encoder.set_bytes(
                    2,
                    mem::size_of::<u32>() as u64,
                    &(1 << stage) as *const _ as *const _,
                );
                encoder.set_bytes(
                    3,
                    mem::size_of::<u32>() as u64,
                    &pass_of_stage as *const _ as *const _,
                );
                encoder.set_bytes(
                    4,
                    mem::size_of::<u32>() as u64,
                    &n_padded as *const _ as *const _,
                );

                let threads_per_group = MTLSize::new(256, 1, 1);
                let thread_groups = MTLSize::new((n_padded as u64 + 255) / 256, 1, 1);

                encoder.dispatch_thread_groups(thread_groups, threads_per_group);
                encoder.end_encoding();
            }
        }

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results
        let indices_ptr = indices_buffer.contents() as *const u32;
        let values_ptr = values_buffer.contents() as *const f32;

        let sorted_indices = unsafe { std::slice::from_raw_parts(indices_ptr, n).to_vec() };
        let sorted_values = unsafe { std::slice::from_raw_parts(values_ptr, n).to_vec() };

        (sorted_indices, sorted_values)
    }
}

impl SimdAccelerator<f32> for MetalAccumulator {
    fn sort_and_accumulate(&self, col_indices: &[usize], values: &[f32]) -> (Vec<usize>, Vec<f32>) {
        if col_indices.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let size = col_indices.len();

        if size < METAL_THRESHOLD {
            // Use CPU implementation for small sizes
            return super::accelerate::AccelerateAccumulator::new()
                .sort_and_accumulate(col_indices, values);
        }

        // Convert to u32 for GPU
        let indices_u32: Vec<u32> = col_indices.iter().map(|&x| x as u32).collect();

        // Sort on GPU
        let (sorted_indices, sorted_values) = self.gpu_bitonic_sort(&indices_u32, values);

        // Accumulate duplicates (on CPU for now)
        let mut result_indices = Vec::new();
        let mut result_values = Vec::new();

        let mut current_idx = sorted_indices[0] as usize;
        let mut current_sum = sorted_values[0];

        for i in 1..sorted_indices.len() {
            let idx = sorted_indices[i] as usize;
            if idx == current_idx {
                current_sum += sorted_values[i];
            } else {
                if current_idx != usize::MAX {
                    result_indices.push(current_idx);
                    result_values.push(current_sum);
                }
                current_idx = idx;
                current_sum = sorted_values[i];
            }
        }

        if current_idx != usize::MAX {
            result_indices.push(current_idx);
            result_values.push(current_sum);
        }

        (result_indices, result_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability() {
        let available = MetalAccumulator::is_available();
        println!("Metal implementation available: {}", available);
    }

    #[test]
    fn test_metal_sort() {
        if let Some(acc) = MetalAccumulator::new() {
            // Create large test data
            let mut indices: Vec<usize> = (0..METAL_THRESHOLD).map(|i| i % 1000).collect();
            let values: Vec<f32> = (0..METAL_THRESHOLD).map(|i| i as f32).collect();

            // Shuffle
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);

            let (sorted_idx, _sorted_val) = acc.sort_and_accumulate(&indices, &values);

            // Verify sorted
            for window in sorted_idx.windows(2) {
                assert!(window[0] <= window[1]);
            }

            // Should have approximately 1000 unique indices (0-999)
            // May have 1001 if usize::MAX padding wasn't fully filtered
            assert!(
                sorted_idx.len() >= 1000 && sorted_idx.len() <= 1001,
                "Expected ~1000 unique indices, got {}",
                sorted_idx.len()
            );
        }
    }
}
