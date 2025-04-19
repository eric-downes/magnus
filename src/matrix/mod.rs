// Matrix data structures and operations

pub mod csr;
pub mod csc;
pub mod config;
pub mod conversion;

pub use csr::SparseMatrixCSR;
pub use csc::SparseMatrixCSC;
pub use config::{MagnusConfig, SystemParameters, Architecture, RowCategory, SortMethod};