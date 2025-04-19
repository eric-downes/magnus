// Matrix data structures and operations

pub mod categorization;
pub mod config;
pub mod conversion;
pub mod csc;
pub mod csr;
pub mod reference;

pub use categorization::{analyze_categorization, categorize_rows, CategorizationSummary};
pub use config::{Architecture, MagnusConfig, RowCategory, SortMethod, SystemParameters};
pub use csc::SparseMatrixCSC;
pub use csr::SparseMatrixCSR;
pub use reference::reference_spgemm;
