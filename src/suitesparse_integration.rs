//! Integration with SuiteSparse Matrix Collection
//!
//! This module provides utilities for loading and working with matrices
//! from the SuiteSparse Matrix Collection (formerly University of Florida
//! Sparse Matrix Collection).

use crate::matrix::SparseMatrixCSR;
use crate::parameter_space::{MatrixGenerator, ParameterConfiguration, PatternMatrixGenerator};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Represents a matrix from the SuiteSparse collection
#[derive(Debug, Clone)]
pub struct SuiteSparseMatrix {
    pub group: String,
    pub name: String,
    pub n_rows: usize,
    pub n_cols: usize,
    pub nnz: usize,
    pub kind: String,
    pub description: String,
}

/// Matrix Market format reader/writer
pub struct MatrixMarketIO;

impl MatrixMarketIO {
    /// Read a matrix in Matrix Market format
    pub fn read_matrix<P: AsRef<Path>>(path: P) -> Result<SparseMatrixCSR<f64>, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Skip comments and read header
        let mut header_line = String::new();
        for line in lines.by_ref() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            if !line.starts_with('%') {
                header_line = line;
                break;
            }
        }

        // Parse header: rows cols nnz
        let parts: Vec<&str> = header_line.split_whitespace().collect();
        if parts.len() != 3 {
            return Err("Invalid Matrix Market header".to_string());
        }

        let n_rows: usize = parts[0]
            .parse()
            .map_err(|_| "Invalid number of rows".to_string())?;
        let n_cols: usize = parts[1]
            .parse()
            .map_err(|_| "Invalid number of columns".to_string())?;
        let nnz: usize = parts[2]
            .parse()
            .map_err(|_| "Invalid number of non-zeros".to_string())?;

        // Read triplets
        let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(nnz);

        for line in lines {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let row: usize = parts[0]
                .parse::<usize>()
                .map_err(|_| "Invalid row index".to_string())?
                .saturating_sub(1); // Convert from 1-indexed to 0-indexed

            let col: usize = parts[1]
                .parse::<usize>()
                .map_err(|_| "Invalid column index".to_string())?
                .saturating_sub(1); // Convert from 1-indexed to 0-indexed

            let val: f64 = if parts.len() >= 3 {
                parts[2]
                    .parse()
                    .map_err(|_| "Invalid value".to_string())?
            } else {
                1.0 // Pattern matrix
            };

            triplets.push((row, col, val));
        }

        // Convert triplets to CSR
        Ok(Self::triplets_to_csr(n_rows, n_cols, triplets))
    }

    /// Write a matrix in Matrix Market format
    pub fn write_matrix<P: AsRef<Path>>(
        path: P,
        matrix: &SparseMatrixCSR<f64>,
    ) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;

        // Write header
        writeln!(
            file,
            "%%MatrixMarket matrix coordinate real general"
        )
        .map_err(|e| format!("Failed to write header: {}", e))?;

        writeln!(
            file,
            "{} {} {}",
            matrix.n_rows,
            matrix.n_cols,
            matrix.nnz()
        )
        .map_err(|e| format!("Failed to write dimensions: {}", e))?;

        // Write non-zeros
        for i in 0..matrix.n_rows {
            let start = matrix.row_ptr[i];
            let end = matrix.row_ptr[i + 1];

            for j in start..end {
                writeln!(
                    file,
                    "{} {} {}",
                    i + 1, // Convert to 1-indexed
                    matrix.col_idx[j] + 1, // Convert to 1-indexed
                    matrix.values[j]
                )
                .map_err(|e| format!("Failed to write entry: {}", e))?;
            }
        }

        Ok(())
    }

    /// Convert triplets to CSR format
    fn triplets_to_csr(
        n_rows: usize,
        n_cols: usize,
        mut triplets: Vec<(usize, usize, f64)>,
    ) -> SparseMatrixCSR<f64> {
        // Sort by row, then column
        triplets.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Combine duplicates
        let mut combined = Vec::new();
        if !triplets.is_empty() {
            let mut current = triplets[0];
            for &(row, col, val) in &triplets[1..] {
                if row == current.0 && col == current.1 {
                    current.2 += val; // Sum duplicates
                } else {
                    combined.push(current);
                    current = (row, col, val);
                }
            }
            combined.push(current);
        }

        // Build CSR
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        let mut current_row = 0;
        for (row, col, val) in combined {
            // Fill empty rows
            while current_row < row {
                row_ptr.push(col_idx.len());
                current_row += 1;
            }

            col_idx.push(col);
            values.push(val);
        }

        // Fill remaining empty rows
        while current_row < n_rows {
            row_ptr.push(col_idx.len());
            current_row += 1;
        }

        SparseMatrixCSR::new(n_rows, n_cols, row_ptr, col_idx, values)
    }
}

/// Generates test matrices similar to those in SuiteSparse collection
pub struct SuiteSparseStyleGenerator {
    matrix_gen: MatrixGenerator,
    pattern_gen: PatternMatrixGenerator,
}

impl SuiteSparseStyleGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            matrix_gen: MatrixGenerator::new(seed),
            pattern_gen: PatternMatrixGenerator::new(seed + 1000),
        }
    }

    /// Generate a matrix similar to circuit simulation matrices
    pub fn generate_circuit_matrix(&mut self, n: usize) -> SparseMatrixCSR<f64> {
        // Circuit matrices typically have:
        // - Small bandwidth
        // - Some dense rows (voltage sources)
        // - Symmetric structure
        let bandwidth = (n as f64).sqrt() as usize;
        self.pattern_gen.generate_banded(n, bandwidth.max(3))
    }

    /// Generate a matrix similar to finite element matrices
    pub fn generate_fem_matrix(&mut self, n: usize) -> SparseMatrixCSR<f64> {
        // FEM matrices typically have:
        // - Block structure from element connectivity
        // - Symmetric positive definite
        let block_size = ((n as f64).sqrt() as usize).max(4);
        self.pattern_gen.generate_block_diagonal(n, block_size)
    }

    /// Generate a matrix similar to web/network graphs
    pub fn generate_web_matrix(&mut self, n: usize) -> SparseMatrixCSR<f64> {
        // Web matrices typically have:
        // - Power law distribution
        // - Very sparse
        // - Highly irregular structure
        self.pattern_gen.generate_power_law(n, 2.1) // Typical web graph exponent
    }

    /// Generate a matrix similar to optimization problems
    pub fn generate_optimization_matrix(&mut self, n: usize) -> SparseMatrixCSR<f64> {
        // Optimization matrices (e.g., LP problems) typically have:
        // - Dense rows (constraints)
        // - Dense columns (variables in many constraints)
        // - Block angular structure
        
        // Create a simple block angular structure
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Dense coupling rows (10% of rows)
        let n_coupling = n / 10;
        for _ in 0..n_coupling {
            // These rows are relatively dense
            for j in 0..n {
                if rng.gen_bool(0.1) {
                    // 10% density in coupling rows
                    col_idx.push(j);
                    values.push(rng.gen_range(-1.0..1.0));
                }
            }
            row_ptr.push(col_idx.len());
        }

        // Block diagonal structure for remaining rows
        let remaining = n - n_coupling;
        let block_size = 50.min(remaining / 4);
        
        for i in 0..remaining {
            let block_id = i / block_size;
            let block_start = n_coupling + block_id * block_size;
            let block_end = (block_start + block_size).min(n);

            for j in block_start..block_end {
                if rng.gen_bool(0.3) {
                    // 30% density within blocks
                    col_idx.push(j);
                    values.push(rng.gen_range(-10.0..10.0));
                }
            }
            row_ptr.push(col_idx.len());
        }

        SparseMatrixCSR::new(n, n, row_ptr, col_idx, values)
    }

    /// Generate a collection of matrices representing different domains
    pub fn generate_collection(&mut self, size: usize) -> Vec<(String, SparseMatrixCSR<f64>)> {
        vec![
            ("circuit".to_string(), self.generate_circuit_matrix(size)),
            ("fem".to_string(), self.generate_fem_matrix(size)),
            ("web".to_string(), self.generate_web_matrix(size)),
            (
                "optimization".to_string(),
                self.generate_optimization_matrix(size),
            ),
        ]
    }
}

/// Downloads and caches matrices from SuiteSparse collection
pub struct SuiteSparseDownloader {
    cache_dir: PathBuf,
}

impl SuiteSparseDownloader {
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir).ok();
        Self { cache_dir }
    }

    /// Get a list of recommended test matrices
    pub fn recommended_matrices() -> Vec<SuiteSparseMatrix> {
        vec![
            SuiteSparseMatrix {
                group: "HB".to_string(),
                name: "bcsstk01".to_string(),
                n_rows: 48,
                n_cols: 48,
                nnz: 224,
                kind: "structural".to_string(),
                description: "Small structural problem".to_string(),
            },
            SuiteSparseMatrix {
                group: "Boeing".to_string(),
                name: "bcsstk13".to_string(),
                n_rows: 2003,
                n_cols: 2003,
                nnz: 83883,
                kind: "structural".to_string(),
                description: "Structural engineering matrix".to_string(),
            },
            SuiteSparseMatrix {
                group: "SNAP".to_string(),
                name: "web-Stanford".to_string(),
                n_rows: 281903,
                n_cols: 281903,
                nnz: 2312497,
                kind: "graph".to_string(),
                description: "Web graph from Stanford".to_string(),
            },
            SuiteSparseMatrix {
                group: "Williams".to_string(),
                name: "mac_econ_fwd500".to_string(),
                n_rows: 206500,
                n_cols: 206500,
                nnz: 1273389,
                kind: "economic".to_string(),
                description: "Economic model matrix".to_string(),
            },
        ]
    }

    /// Check if a matrix is cached
    pub fn is_cached(&self, matrix: &SuiteSparseMatrix) -> bool {
        self.get_cache_path(matrix).exists()
    }

    /// Get the cache path for a matrix
    fn get_cache_path(&self, matrix: &SuiteSparseMatrix) -> PathBuf {
        self.cache_dir
            .join(format!("{}_{}.mtx", matrix.group, matrix.name))
    }

    /// Note: Actual downloading would require HTTP client
    /// This is a placeholder for the download functionality
    pub fn download_matrix(&self, matrix: &SuiteSparseMatrix) -> Result<PathBuf, String> {
        let cache_path = self.get_cache_path(matrix);
        
        if cache_path.exists() {
            return Ok(cache_path);
        }

        // In a real implementation, this would download from:
        // https://sparse.tamu.edu/MM/{group}/{name}.tar.gz
        // and extract the .mtx file
        
        Err(format!(
            "Downloading from SuiteSparse collection requires external HTTP client. \
             Please download {}/{} manually from https://sparse.tamu.edu/",
            matrix.group, matrix.name
        ))
    }

    /// Load a cached matrix
    pub fn load_matrix(&self, matrix: &SuiteSparseMatrix) -> Result<SparseMatrixCSR<f64>, String> {
        let path = self.get_cache_path(matrix);
        if !path.exists() {
            return Err(format!("Matrix {} not found in cache", matrix.name));
        }
        MatrixMarketIO::read_matrix(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_matrix_market_io() {
        // Create a simple test matrix
        let matrix = SparseMatrixCSR::new(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        );

        // Write to temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();
        
        MatrixMarketIO::write_matrix(&path, &matrix).unwrap();

        // Read back
        let loaded = MatrixMarketIO::read_matrix(&path).unwrap();

        assert_eq!(loaded.n_rows, matrix.n_rows);
        assert_eq!(loaded.n_cols, matrix.n_cols);
        assert_eq!(loaded.nnz(), matrix.nnz());
    }

    #[test]
    fn test_suitesparse_style_generation() {
        let mut gen = SuiteSparseStyleGenerator::new(42);

        let circuit = gen.generate_circuit_matrix(100);
        assert_eq!(circuit.n_rows, 100);
        println!("Circuit matrix nnz: {}", circuit.nnz());

        let fem = gen.generate_fem_matrix(100);
        assert_eq!(fem.n_rows, 100);
        println!("FEM matrix nnz: {}", fem.nnz());

        let web = gen.generate_web_matrix(100);
        assert_eq!(web.n_rows, 100);
        println!("Web matrix nnz: {}", web.nnz());

        let opt = gen.generate_optimization_matrix(100);
        assert_eq!(opt.n_rows, 100);
        println!("Optimization matrix nnz: {}", opt.nnz());
    }

    #[test]
    fn test_matrix_market_format() {
        // Create a Matrix Market format string
        let mtx_content = "%%MatrixMarket matrix coordinate real general\n\
                          3 3 5\n\
                          1 1 1.0\n\
                          1 3 2.0\n\
                          2 2 3.0\n\
                          3 1 4.0\n\
                          3 3 5.0\n";

        // Write to temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", mtx_content).unwrap();

        // Read the matrix
        let matrix = MatrixMarketIO::read_matrix(temp_file.path()).unwrap();

        assert_eq!(matrix.n_rows, 3);
        assert_eq!(matrix.n_cols, 3);
        assert_eq!(matrix.nnz(), 5);

        // Check values
        assert_eq!(matrix.row_ptr, vec![0, 2, 3, 5]);
        assert_eq!(matrix.col_idx, vec![0, 2, 1, 0, 2]);
        assert_eq!(matrix.values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}