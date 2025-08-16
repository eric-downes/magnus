//! Zero-cost duplicate rate prediction for adaptive accumulation
//!
//! This module provides prediction of duplicate rates using information
//! already available during SpGEMM computation, with no additional overhead.

/// Context for duplicate rate prediction
#[derive(Debug, Clone, Copy)]
pub struct DuplicateContext {
    /// Number of columns in matrix B
    pub b_ncols: usize,
    /// Expected number of products for this row
    pub expected_products: usize,
}

impl DuplicateContext {
    /// Create new context from SpGEMM parameters
    pub fn new(b_ncols: usize) -> Self {
        Self {
            b_ncols,
            expected_products: 0,
        }
    }
    
    /// Update expected products count (called during row traversal)
    #[inline]
    pub fn add_products(&mut self, count: usize) {
        self.expected_products += count;
    }
    
    /// Predict whether this row will have high duplicate rate
    /// Returns true if expected duplicates > 75%
    #[inline]
    pub fn predicts_high_duplicates(&self) -> bool {
        // Use 4x threshold: if products > 4 * columns, expect >75% duplicates
        self.expected_products > self.b_ncols * 4
    }
    
    /// Get expected duplicate rate (0.0 to 1.0)
    #[inline]
    pub fn expected_duplicate_rate(&self) -> f64 {
        if self.expected_products <= self.b_ncols {
            0.0
        } else {
            1.0 - (self.b_ncols as f64 / self.expected_products as f64)
        }
    }
}

/// Strategy for accumulation based on predicted duplicates
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccumulationStrategy {
    /// Use platform-optimized sort (Accelerate/NEON) then accumulate
    Separated,
    /// Use fused sort+accumulate for high duplicates
    Fused,
}

impl AccumulationStrategy {
    /// Select strategy based on duplicate context
    #[inline]
    pub fn from_context(ctx: &DuplicateContext) -> Self {
        if ctx.predicts_high_duplicates() {
            Self::Fused
        } else {
            Self::Separated
        }
    }
}

/// Zero-cost duplicate predictor for SpGEMM rows
pub struct DuplicatePredictor {
    b_ncols: usize,
    b_row_sizes: Vec<usize>,
}

impl DuplicatePredictor {
    /// Create predictor from matrix B structure
    pub fn from_matrix_b<T>(b_row_ptr: &[usize], b_ncols: usize) -> Self {
        // Precompute row sizes for O(1) lookup
        let b_row_sizes: Vec<usize> = b_row_ptr.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        Self {
            b_ncols,
            b_row_sizes,
        }
    }
    
    /// Predict duplicates for a row multiplication (zero additional cost)
    /// This uses data we're already accessing in SpGEMM
    #[inline]
    pub fn predict_for_row(&self, a_col_indices: &[usize]) -> DuplicateContext {
        let mut ctx = DuplicateContext::new(self.b_ncols);
        
        // Sum up expected products from B rows
        // We're already iterating these indices in SpGEMM!
        for &k in a_col_indices {
            if k < self.b_row_sizes.len() {
                ctx.add_products(self.b_row_sizes[k]);
            }
        }
        
        ctx
    }
    
    /// Quick check if row will have high duplicates
    #[inline]
    pub fn quick_check(&self, a_row_nnz: usize, avg_b_row_nnz: usize) -> bool {
        // Simple heuristic: products > 4 * columns
        a_row_nnz * avg_b_row_nnz > self.b_ncols * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_duplicate_context() {
        let mut ctx = DuplicateContext::new(100);
        
        // Low products: no duplicates expected
        ctx.add_products(50);
        assert!(!ctx.predicts_high_duplicates());
        assert!(ctx.expected_duplicate_rate() < 0.1);
        
        // High products: many duplicates expected
        ctx.add_products(450); // Total 500 products for 100 columns
        assert!(ctx.predicts_high_duplicates());
        assert!(ctx.expected_duplicate_rate() > 0.75);
    }
    
    #[test]
    fn test_threshold_boundary() {
        let ctx_low = DuplicateContext {
            b_ncols: 100,
            expected_products: 399, // Just under 4x
        };
        assert!(!ctx_low.predicts_high_duplicates());
        
        let ctx_high = DuplicateContext {
            b_ncols: 100,
            expected_products: 401, // Just over 4x
        };
        assert!(ctx_high.predicts_high_duplicates());
    }
    
    #[test]
    fn test_predictor() {
        // Matrix B with varying row sizes
        let b_row_ptr = vec![0, 10, 30, 35, 100]; // Rows with 10, 20, 5, 65 nnz
        let predictor = DuplicatePredictor::from_matrix_b::<f32>(&b_row_ptr, 50);
        
        // Row of A pointing to dense rows of B
        let a_cols = vec![1, 3]; // Points to rows with 20 and 65 nnz
        let ctx = predictor.predict_for_row(&a_cols);
        
        assert_eq!(ctx.expected_products, 85); // 20 + 65
        assert!(ctx.expected_duplicate_rate() > 0.4); // 85 products into 50 columns
    }
}