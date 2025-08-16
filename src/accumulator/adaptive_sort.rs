//! Adaptive sort accumulator that chooses between fused and separated strategies
//!
//! This module provides an enhanced sort accumulator that uses zero-cost
//! duplicate prediction to choose the optimal accumulation strategy.

use num_traits::Num;
use std::ops::AddAssign;
use super::duplicate_prediction::{DuplicateContext, AccumulationStrategy};
use super::Accumulator;

/// Adaptive sort-based accumulator with strategy selection
pub struct AdaptiveSortAccumulator<T> {
    /// Column indices of intermediate products
    col_indices: Vec<usize>,
    /// Values of intermediate products
    values: Vec<T>,
    /// Context for duplicate prediction
    context: Option<DuplicateContext>,
}

impl<T> AdaptiveSortAccumulator<T>
where
    T: Copy + Num + AddAssign,
{
    /// Create new accumulator with initial capacity
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            col_indices: Vec::with_capacity(initial_capacity),
            values: Vec::with_capacity(initial_capacity),
            context: None,
        }
    }
    
    /// Create with duplicate context for strategy selection
    pub fn with_context(initial_capacity: usize, context: DuplicateContext) -> Self {
        Self {
            col_indices: Vec::with_capacity(initial_capacity),
            values: Vec::with_capacity(initial_capacity),
            context: Some(context),
        }
    }
    
    /// Set context for duplicate prediction (zero-cost)
    pub fn set_context(&mut self, context: DuplicateContext) {
        self.context = Some(context);
    }
}

impl<T> Accumulator<T> for AdaptiveSortAccumulator<T>
where
    T: Copy + Num + AddAssign + 'static,
{
    fn reset(&mut self) {
        self.col_indices.clear();
        self.values.clear();
        // Keep context for next row
    }
    
    fn accumulate(&mut self, col: usize, val: T) {
        self.col_indices.push(col);
        self.values.push(val);
    }
    
    fn extract_result(self) -> (Vec<usize>, Vec<T>) {
        if self.col_indices.is_empty() {
            return (Vec::new(), Vec::new());
        }
        
        // Decide strategy based on context
        let strategy = self.context
            .map(|ctx| AccumulationStrategy::from_context(&ctx))
            .unwrap_or(AccumulationStrategy::Separated);
        
        match strategy {
            AccumulationStrategy::Fused => {
                // Use fused implementation for high duplicates
                extract_fused(self.col_indices, self.values)
            }
            AccumulationStrategy::Separated => {
                // Use platform-optimized sort + separate accumulation
                extract_separated(self.col_indices, self.values)
            }
        }
    }
}

/// Extract using fused sort+accumulate (for high duplicates)
fn extract_fused<T>(col_indices: Vec<usize>, values: Vec<T>) -> (Vec<usize>, Vec<T>)
where
    T: Copy + Num + AddAssign + 'static,
{
    // For f32, use our optimized implementation
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Safe transmute for f32
        let indices = col_indices;
        let vals = unsafe {
            std::mem::transmute::<Vec<T>, Vec<f32>>(values)
        };
        
        let (result_idx, result_val) = super::fused::fused_sort_accumulate(indices, vals);
        
        let result_values = unsafe {
            std::mem::transmute::<Vec<f32>, Vec<T>>(result_val)
        };
        
        return (result_idx, result_values);
    }
    
    // Generic implementation
    super::fused::fused_sort_accumulate_generic(col_indices, values)
}

/// Extract using separated sort then accumulate (default)
fn extract_separated<T>(col_indices: Vec<usize>, values: Vec<T>) -> (Vec<usize>, Vec<T>)
where
    T: Copy + Num + AddAssign + 'static,
{
    // For f32 on Apple Silicon, use Accelerate
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            use super::SimdAccelerator;
            
            let vals = unsafe {
                std::mem::transmute::<Vec<T>, Vec<f32>>(values.clone())
            };
            
            let acc = super::accelerate::AccelerateAccumulator::new();
            let (result_idx, result_val) = acc.sort_and_accumulate(&col_indices, &vals);
            
            let result_values = unsafe {
                std::mem::transmute::<Vec<f32>, Vec<T>>(result_val)
            };
            
            return (result_idx, result_values);
        }
    }
    
    // Fallback: standard sort then accumulate
    let mut pairs: Vec<_> = col_indices.into_iter()
        .zip(values.into_iter())
        .collect();
    
    pairs.sort_unstable_by_key(|&(idx, _)| idx);
    
    // Accumulate duplicates
    let mut result_indices = Vec::new();
    let mut result_values = Vec::new();
    
    if !pairs.is_empty() {
        let mut current_idx = pairs[0].0;
        let mut current_val = pairs[0].1;
        
        for &(idx, val) in &pairs[1..] {
            if idx == current_idx {
                current_val += val;
            } else {
                result_indices.push(current_idx);
                result_values.push(current_val);
                current_idx = idx;
                current_val = val;
            }
        }
        
        result_indices.push(current_idx);
        result_values.push(current_val);
    }
    
    (result_indices, result_values)
}

/// Create adaptive accumulator with automatic context
pub fn create_adaptive_accumulator<T>(
    initial_capacity: usize,
    b_ncols: Option<usize>,
    expected_products: Option<usize>,
) -> Box<dyn Accumulator<T>>
where
    T: Copy + Num + AddAssign + 'static,
{
    let mut acc = AdaptiveSortAccumulator::new(initial_capacity);
    
    if let (Some(ncols), Some(products)) = (b_ncols, expected_products) {
        let mut context = DuplicateContext::new(ncols);
        context.add_products(products);
        acc.set_context(context);
    }
    
    Box::new(acc)
}

// We need to import TypeId
use std::any::TypeId;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptive_low_duplicates() {
        // Low duplicate scenario - should use separated strategy
        let context = DuplicateContext {
            b_ncols: 100,
            expected_products: 200, // 2x ratio, ~50% duplicates
        };
        
        let mut acc = AdaptiveSortAccumulator::with_context(10, context);
        
        acc.accumulate(5, 1.0);
        acc.accumulate(3, 2.0);
        acc.accumulate(5, 3.0); // Duplicate
        acc.accumulate(7, 4.0);
        
        let (indices, values) = acc.extract_result();
        
        assert_eq!(indices, vec![3, 5, 7]);
        assert_eq!(values, vec![2.0, 4.0, 4.0]);
    }
    
    #[test]
    fn test_adaptive_high_duplicates() {
        // High duplicate scenario - should use fused strategy
        let context = DuplicateContext {
            b_ncols: 10,
            expected_products: 50, // 5x ratio, ~80% duplicates
        };
        
        let mut acc = AdaptiveSortAccumulator::with_context(10, context);
        
        // Add many duplicates
        for i in 0..50 {
            acc.accumulate(i % 10, i as f32);
        }
        
        let (indices, values) = acc.extract_result();
        
        assert_eq!(indices.len(), 10); // Should have 10 unique indices
        
        // Check that accumulation happened correctly
        for i in 0..10 {
            // Sum of 0, 10, 20, 30, 40 = 100
            // Sum of 1, 11, 21, 31, 41 = 105
            // etc.
            let expected_sum: f32 = (0..5).map(|j| (i + j * 10) as f32).sum();
            assert_eq!(values[i], expected_sum);
        }
    }
}