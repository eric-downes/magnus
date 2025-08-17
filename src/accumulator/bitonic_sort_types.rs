//! Type-safe abstractions for bitonic sort algorithm
//!
//! This module provides structural types that enforce the invariants
//! required by the bitonic sort algorithm, making the logic more
//! explicit and less reliant on clever bit manipulation tricks.

use std::marker::PhantomData;

/// A power of two value, guaranteed at the type level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerOfTwo {
    value: u32,
    log2_value: u32,
}

impl PowerOfTwo {
    /// Create a new PowerOfTwo from a value
    /// Returns None if the value is not a power of two
    pub fn new(value: u32) -> Option<Self> {
        if value == 0 || (value & (value - 1)) != 0 {
            None
        } else {
            Some(PowerOfTwo {
                value,
                log2_value: value.trailing_zeros(),
            })
        }
    }

    /// Create from a log2 value (e.g., 3 creates 8)
    pub fn from_log2(log2_value: u32) -> Self {
        PowerOfTwo {
            value: 1 << log2_value,
            log2_value,
        }
    }

    /// Get the raw value
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Get the log2 of the value
    pub fn log2(&self) -> u32 {
        self.log2_value
    }

    /// Get the next power of two >= n
    pub fn next_power_of_two(n: usize) -> Self {
        let value = n.next_power_of_two() as u32;
        PowerOfTwo::new(value).unwrap()
    }
}

/// Represents a stage in the bitonic sort algorithm
#[derive(Debug, Clone, Copy)]
pub struct BitonicStage {
    stage_number: u32,
    stage_size: PowerOfTwo,
}

impl BitonicStage {
    /// Create a new stage (0-indexed)
    pub fn new(stage_number: u32) -> Self {
        BitonicStage {
            stage_number,
            stage_size: PowerOfTwo::from_log2(stage_number),
        }
    }

    /// Get the stage number (0-indexed)
    pub fn number(&self) -> u32 {
        self.stage_number
    }

    /// Get the size of comparisons at this stage
    pub fn size(&self) -> PowerOfTwo {
        self.stage_size
    }

    /// Get the number of passes in this stage
    pub fn num_passes(&self) -> u32 {
        self.stage_number + 1
    }
}

/// Represents a pass within a stage of bitonic sort
#[derive(Debug, Clone, Copy)]
pub struct BitonicPass {
    stage: BitonicStage,
    pass_index: u32,
    pass_distance: PowerOfTwo,
}

impl BitonicPass {
    /// Create a new pass within a stage
    pub fn new(stage: BitonicStage, pass_index: u32) -> Option<Self> {
        if pass_index > stage.number() {
            None
        } else {
            let pass_distance = PowerOfTwo::from_log2(stage.number() - pass_index);
            Some(BitonicPass {
                stage,
                pass_index,
                pass_distance,
            })
        }
    }

    /// Get the comparison distance for this pass
    pub fn distance(&self) -> PowerOfTwo {
        self.pass_distance
    }

    /// Get the block size for determining sort direction
    pub fn block_size(&self) -> PowerOfTwo {
        PowerOfTwo::from_log2(self.pass_distance.log2() + 1)
    }
}

/// Thread context for bitonic sort
#[derive(Debug, Clone, Copy)]
pub struct BitonicThread {
    thread_id: u32,
}

impl BitonicThread {
    pub fn new(thread_id: u32) -> Self {
        BitonicThread { thread_id }
    }

    /// Find the partner thread for comparison in this pass
    pub fn partner(&self, pass: &BitonicPass) -> u32 {
        self.thread_id ^ pass.distance().value()
    }

    /// Determine if this thread should sort in ascending order
    pub fn is_ascending(&self, pass: &BitonicPass) -> bool {
        let block_size = pass.block_size().value();
        (self.thread_id & block_size) == 0
    }

    /// Check if this thread should perform a comparison in this pass
    pub fn should_compare(&self, pass: &BitonicPass, n_elements: u32) -> bool {
        let partner = self.partner(pass);
        partner > self.thread_id && partner < n_elements
    }

    pub fn id(&self) -> u32 {
        self.thread_id
    }
}

/// Bitonic sort configuration
#[derive(Debug, Clone)]
pub struct BitonicSortConfig {
    n_elements: usize,
    padded_size: PowerOfTwo,
    num_stages: u32,
}

impl BitonicSortConfig {
    pub fn new(n_elements: usize) -> Self {
        let padded_size = PowerOfTwo::next_power_of_two(n_elements);
        let num_stages = padded_size.log2();
        
        BitonicSortConfig {
            n_elements,
            padded_size,
            num_stages,
        }
    }

    pub fn padded_size(&self) -> usize {
        self.padded_size.value() as usize
    }

    pub fn num_stages(&self) -> u32 {
        self.num_stages
    }

    pub fn original_size(&self) -> usize {
        self.n_elements
    }

    /// Iterate over all stages
    pub fn stages(&self) -> impl Iterator<Item = BitonicStage> {
        (0..self.num_stages).map(BitonicStage::new)
    }

    /// Iterate over all passes for a given stage
    pub fn passes(&self, stage: BitonicStage) -> impl Iterator<Item = BitonicPass> {
        (0..=stage.number()).filter_map(move |i| BitonicPass::new(stage, i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_of_two() {
        assert!(PowerOfTwo::new(0).is_none());
        assert!(PowerOfTwo::new(3).is_none());
        assert!(PowerOfTwo::new(5).is_none());
        
        let p4 = PowerOfTwo::new(4).unwrap();
        assert_eq!(p4.value(), 4);
        assert_eq!(p4.log2(), 2);
        
        let p16 = PowerOfTwo::from_log2(4);
        assert_eq!(p16.value(), 16);
        assert_eq!(p16.log2(), 4);
    }

    #[test]
    fn test_bitonic_thread_direction() {
        let stage = BitonicStage::new(2); // Stage 2
        let pass = BitonicPass::new(stage, 0).unwrap(); // First pass of stage 2
        
        // Block size should be 8 for this pass
        assert_eq!(pass.block_size().value(), 8);
        
        // Threads 0-7 should sort ascending
        for tid in 0..8 {
            let thread = BitonicThread::new(tid);
            assert!(thread.is_ascending(&pass));
        }
        
        // Threads 8-15 should sort descending
        for tid in 8..16 {
            let thread = BitonicThread::new(tid);
            assert!(!thread.is_ascending(&pass));
        }
    }

    #[test]
    fn test_config_iteration() {
        let config = BitonicSortConfig::new(10);
        assert_eq!(config.padded_size(), 16);
        assert_eq!(config.num_stages(), 4);
        
        let stages: Vec<_> = config.stages().collect();
        assert_eq!(stages.len(), 4);
        
        for stage in stages {
            let passes: Vec<_> = config.passes(stage).collect();
            assert_eq!(passes.len() as u32, stage.num_passes());
        }
    }
}