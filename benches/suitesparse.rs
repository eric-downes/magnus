//! Benchmarks with SuiteSparse Matrix Collection

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use magnus::{
    multiply_row_coarse_level, multiply_row_fine_level, reference_spgemm, MagnusConfig,
    SparseMatrixCSR,
};
use std::fs;
use std::path::Path;

/// Benchmark on SuiteSparse matrices
fn bench_suitesparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("SuiteSparse");

    // Define paths to SuiteSparse matrices
    // These would need to be downloaded separately
    let suitesparse_dir = Path::new("./data/suitesparse");

    // Check if the directory exists
    if !suitesparse_dir.exists() {
        eprintln!(
            "SuiteSparse directory not found. Create ./data/suitesparse and add matrices there."
        );
        return;
    }

    // List of SuiteSparse matrices to benchmark
    // These are from Table 2 in the MAGNUS paper
    let matrix_names = ["cage4", "bcsstk08", "bcsstk09", "west0067", "dwt_234"];

    let config = MagnusConfig::default();

    for &name in &matrix_names {
        let matrix_path = suitesparse_dir.join(format!("{}.mtx", name));

        if !matrix_path.exists() {
            eprintln!("Matrix file not found: {:?}", matrix_path);
            continue;
        }

        // Load the matrix from file
        match load_matrix_market(&matrix_path) {
            Ok(a) => {
                // Create a compatible B matrix (for now, just use A)
                // In practice, we would use specific B matrices for each test
                let b = a.clone();

                // Benchmark reference multiplication
                group.bench_with_input(
                    BenchmarkId::new(format!("reference_{}", name), name),
                    &name,
                    |bench, _| {
                        bench.iter(|| {
                            let c = reference_spgemm(&a, &b);
                            black_box(c)
                        })
                    },
                );

                // Benchmark selected rows using fine-level reordering
                if let Some(row_idx) = select_representative_row(&a, RowType::Medium) {
                    group.bench_with_input(
                        BenchmarkId::new(format!("fine_level_{}", name), name),
                        &name,
                        |bench, _| {
                            bench.iter(|| {
                                let (cols, vals) =
                                    multiply_row_fine_level(row_idx, &a, &b, &config);
                                black_box((cols, vals))
                            })
                        },
                    );
                }

                // Benchmark selected rows using coarse-level reordering
                if let Some(row_idx) = select_representative_row(&a, RowType::Large) {
                    group.bench_with_input(
                        BenchmarkId::new(format!("coarse_level_{}", name), name),
                        &name,
                        |bench, _| {
                            bench.iter(|| {
                                let (cols, vals) =
                                    multiply_row_coarse_level(row_idx, &a, &b, &config);
                                black_box((cols, vals))
                            })
                        },
                    );
                }
            }
            Err(err) => {
                eprintln!("Error loading matrix {}: {}", name, err);
            }
        }
    }

    group.finish();
}

/// The type of row to select for benchmarking
enum RowType {
    // Small variant removed as it's not currently used
    Medium, // Medium number of intermediate products
    Large,  // Many intermediate products
}

/// Select a representative row of the given type from the matrix
fn select_representative_row(a: &SparseMatrixCSR<f64>, row_type: RowType) -> Option<usize> {
    let n_rows = a.n_rows;
    if n_rows == 0 {
        return None;
    }

    // Calculate the number of non-zeros in each row
    let mut row_nnz: Vec<(usize, usize)> = (0..n_rows)
        .map(|i| {
            let nnz = a.row_ptr[i + 1] - a.row_ptr[i];
            (i, nnz)
        })
        .collect();

    // Sort rows by number of non-zeros
    row_nnz.sort_by_key(|&(_, nnz)| nnz);

    // Select a row based on the requested type
    match row_type {
        RowType::Medium => {
            // Pick a row in the middle by non-zeros
            let idx = n_rows / 2;
            Some(row_nnz[idx].0)
        }
        RowType::Large => {
            // Pick a row in the top 20% by non-zeros
            let idx = (n_rows * 4) / 5;
            Some(row_nnz[idx].0)
        }
    }
}

/// Load a matrix from Matrix Market format
fn load_matrix_market(path: &Path) -> Result<SparseMatrixCSR<f64>, String> {
    // This is a simple implementation - in practice, you might use a library
    // like `sprs` that can load Matrix Market files directly

    let contents = fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    let mut lines = contents.lines();

    // Skip header and comments
    let mut header_line = lines.next().ok_or("Empty file")?;
    while header_line.starts_with("%") {
        header_line = lines.next().ok_or("No data after comments")?;
    }

    // Parse matrix dimensions
    let dims: Vec<usize> = header_line
        .split_whitespace()
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();

    if dims.len() < 3 {
        return Err("Invalid header format".to_string());
    }

    let rows = dims[0];
    let cols = dims[1];
    let nnz = dims[2];

    // Parse matrix entries
    let mut entries = Vec::with_capacity(nnz);

    for line in lines {
        let values: Vec<&str> = line.split_whitespace().collect();
        if values.len() < 3 {
            continue; // Skip invalid lines
        }

        let row = values[0]
            .parse::<usize>()
            .map_err(|_| "Invalid row index")?
            - 1; // 1-indexed to 0-indexed
        let col = values[1]
            .parse::<usize>()
            .map_err(|_| "Invalid column index")?
            - 1; // 1-indexed to 0-indexed
        let val = values[2].parse::<f64>().map_err(|_| "Invalid value")?;

        if row < rows && col < cols {
            entries.push((row, col, val));
        }
    }

    // Sort entries by row, then column
    entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Build CSR format
    let mut row_ptr = vec![0; rows + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    let mut current_row = 0;

    for (row, col, val) in entries {
        // Update row pointers if we've moved to a new row
        while current_row < row {
            current_row += 1;
            row_ptr[current_row] = col_idx.len();
        }

        col_idx.push(col);
        values.push(val);
    }

    // Fill remaining row pointers
    for i in (current_row + 1)..=rows {
        row_ptr[i] = col_idx.len();
    }

    Ok(SparseMatrixCSR::new(rows, cols, row_ptr, col_idx, values))
}

criterion_group!(benches, bench_suitesparse);
criterion_main!(benches);
