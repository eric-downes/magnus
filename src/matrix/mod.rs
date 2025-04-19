// Matrix data structures and operations

pub mod csr;
pub mod csc;
pub mod config;
pub mod conversion;
pub mod reference;
pub mod categorization;

pub use csr::SparseMatrixCSR;
pub use csc::SparseMatrixCSC;
pub use config::{MagnusConfig, SystemParameters, Architecture, RowCategory, SortMethod};
pub use reference::reference_spgemm;
pub use categorization::{categorize_rows, analyze_categorization, CategorizationSummary};