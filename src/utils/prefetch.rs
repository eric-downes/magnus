//! Memory prefetching utilities for optimizing cache performance
//!
//! This module provides architecture-specific prefetch hints to improve
//! memory access patterns in sparse matrix operations.

use std::ptr;

/// Prefetch strategy for memory access optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Conservative: Only prefetch next row (minimal overhead)
    Conservative,
    /// Moderate: Prefetch next row + likely B matrix rows
    Moderate,
    /// Aggressive: Full lookahead prefetching
    Aggressive,
    /// Adaptive: Adjust based on matrix characteristics
    Adaptive,
}

/// Configuration for memory prefetching
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Enable prefetching
    pub enabled: bool,
    /// Prefetch distance (how many elements ahead)
    pub distance: usize,
    /// Prefetch strategy
    pub strategy: PrefetchStrategy,
    /// Memory threshold for auto-enable (bytes)
    pub memory_threshold: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

impl PrefetchConfig {
    /// Create configuration with no prefetching
    pub fn none() -> Self {
        Self {
            enabled: false,
            distance: 0,
            strategy: PrefetchStrategy::None,
            memory_threshold: 0,
        }
    }
    
    /// Create conservative prefetch configuration
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            distance: 1,
            strategy: PrefetchStrategy::Conservative,
            memory_threshold: 2 * 1024 * 1024 * 1024, // 2GB
        }
    }
    
    /// Create moderate prefetch configuration
    pub fn moderate() -> Self {
        Self {
            enabled: true,
            distance: 4,
            strategy: PrefetchStrategy::Moderate,
            memory_threshold: 4 * 1024 * 1024 * 1024, // 4GB
        }
    }
    
    /// Create aggressive prefetch configuration
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            distance: 8,
            strategy: PrefetchStrategy::Aggressive,
            memory_threshold: 8 * 1024 * 1024 * 1024, // 8GB
        }
    }
    
    /// Auto-detect optimal configuration based on system
    pub fn auto_detect() -> Self {
        // Check environment variable first
        if let Ok(strategy) = std::env::var("MAGNUS_PREFETCH") {
            return match strategy.to_lowercase().as_str() {
                "none" => Self::none(),
                "conservative" => Self::conservative(),
                "moderate" => Self::moderate(),
                "aggressive" => Self::aggressive(),
                _ => Self::default_for_system(),
            };
        }
        
        Self::default_for_system()
    }
    
    /// Get default configuration based on system memory
    fn default_for_system() -> Self {
        let total_memory = get_system_memory();
        
        if total_memory < 4 * 1024 * 1024 * 1024 {
            // Less than 4GB: Conservative or none
            Self::conservative()
        } else if total_memory < 8 * 1024 * 1024 * 1024 {
            // 4-8GB: Moderate
            Self::moderate()
        } else {
            // 8GB+: Can be aggressive
            Self::moderate() // Still default to moderate for safety
        }
    }
}

/// Get total system memory in bytes
fn get_system_memory() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        
        if let Ok(output) = Command::new("sysctl")
            .args(&["-n", "hw.memsize"])
            .output() {
            let mem_str = String::from_utf8_lossy(&output.stdout);
            mem_str.trim().parse().unwrap_or(8 * 1024 * 1024 * 1024)
        } else {
            8 * 1024 * 1024 * 1024
        }
    }
    
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: usize = parts[1].parse().unwrap_or(8 * 1024 * 1024);
                        return kb * 1024; // Convert KB to bytes
                    }
                }
            }
        }
        8 * 1024 * 1024 * 1024
    }
    
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        8 * 1024 * 1024 * 1024 // Default to 8GB
    }
}

// Architecture-specific prefetch instructions

/// Prefetch for read into L1 cache
#[inline(always)]
pub fn prefetch_read_l1<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PLDL1KEEP, [ptr]
        // Prefetch for load, L1, temporal (keep in cache)
        // Note: _prefetch is unstable, use inline assembly instead
        std::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // T0 hint: Prefetch into all cache levels
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        // No-op on unsupported architectures
        let _ = ptr;
    }
}

/// Prefetch for read into L2 cache
#[inline(always)]
pub fn prefetch_read_l2<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PLDL2KEEP, [ptr]
        // Prefetch for load, L2, temporal
        std::arch::asm!(
            "prfm pldl2keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // T1 hint: Prefetch into L2 and above
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T1);
    }
    
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch for write
#[inline(always)]
pub fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PSTL1KEEP, [ptr]
        // Prefetch for store, L1, temporal
        std::arch::asm!(
            "prfm pstl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // x86 doesn't have separate write prefetch, use T0
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

/// Non-temporal prefetch (streaming, don't keep in cache)
#[inline(always)]
pub fn prefetch_non_temporal<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PLDL1STRM, [ptr]
        // Prefetch for load, L1, streaming (use once)
        std::arch::asm!(
            "prfm pldl1strm, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // NTA hint: Non-temporal, bypass cache if possible
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_NTA);
    }
    
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch multiple cache lines ahead
#[inline(always)]
pub fn prefetch_range<T>(start_ptr: *const T, count: usize, stride: usize) {
    let cache_line_size = 64; // Typical cache line size
    let element_size = std::mem::size_of::<T>();
    let elements_per_line = cache_line_size / element_size.max(1);
    
    for i in (0..count).step_by(elements_per_line) {
        unsafe {
            let ptr = start_ptr.add(i * stride);
            prefetch_read_l1(ptr);
        }
    }
}

/// Memory access pattern analyzer for adaptive prefetching
pub struct AccessPatternAnalyzer {
    hits: usize,
    misses: usize,
    total_accesses: usize,
}

impl AccessPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            total_accesses: 0,
        }
    }
    
    /// Record a memory access (simplified - real implementation would use perf counters)
    pub fn record_access(&mut self, was_hit: bool) {
        self.total_accesses += 1;
        if was_hit {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
    }
    
    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_accesses as f64
        }
    }
    
    /// Recommend prefetch strategy based on observed patterns
    pub fn recommend_strategy(&self) -> PrefetchStrategy {
        let hit_rate = self.hit_rate();
        
        if hit_rate > 0.9 {
            // Excellent cache behavior, minimal prefetch needed
            PrefetchStrategy::Conservative
        } else if hit_rate > 0.7 {
            // Good cache behavior, moderate prefetch helpful
            PrefetchStrategy::Moderate
        } else {
            // Poor cache behavior, aggressive prefetch may help
            PrefetchStrategy::Aggressive
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prefetch_config() {
        let config = PrefetchConfig::conservative();
        assert!(config.enabled);
        assert_eq!(config.strategy, PrefetchStrategy::Conservative);
        
        let config = PrefetchConfig::none();
        assert!(!config.enabled);
    }
    
    #[test]
    fn test_prefetch_instructions() {
        // Just ensure they compile and don't crash
        let data = vec![1, 2, 3, 4, 5];
        let ptr = data.as_ptr();
        
        prefetch_read_l1(ptr);
        prefetch_read_l2(ptr);
        prefetch_non_temporal(ptr);
        
        let mut data_mut = vec![1, 2, 3, 4, 5];
        let ptr_mut = data_mut.as_mut_ptr();
        prefetch_write(ptr_mut);
    }
    
    #[test]
    fn test_pattern_analyzer() {
        let mut analyzer = AccessPatternAnalyzer::new();
        
        // Simulate good cache behavior
        for _ in 0..90 {
            analyzer.record_access(true); // hit
        }
        for _ in 0..10 {
            analyzer.record_access(false); // miss
        }
        
        assert!(analyzer.hit_rate() > 0.85);
        assert_eq!(analyzer.recommend_strategy(), PrefetchStrategy::Conservative);
    }
}