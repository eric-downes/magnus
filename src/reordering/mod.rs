//! Reordering algorithms for improving data locality

// This is a placeholder for now.
// Will be implemented in Phase 2 of the roadmap.

/// Metadata for chunking operations
pub struct ChunkMetadata {
    /// Size of each chunk
    pub chunk_length: usize,
    
    /// Number of chunks
    pub n_chunks: usize,
    
    /// Number of bits to shift right to get chunk index (for power-of-2 chunk sizes)
    pub shift_bits: usize,
}