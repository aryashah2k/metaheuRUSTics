//! # Metaheuristics
//! 
//! A comprehensive collection of metaheuristic optimization algorithms implemented in Rust.
//! This library provides implementations of various optimization algorithms for solving
//! complex optimization problems.

pub mod algorithm;
pub mod test_function;
pub mod utils;

/// Common types and traits used throughout the library
pub mod prelude {
    pub use crate::algorithm::*;
    pub use crate::test_function::*;
    pub use crate::utils::*;
}

/// Error types for the metaheuristics library
#[derive(thiserror::Error, Debug)]
pub enum MetaheuristicError {
    #[error("Invalid dimension: expected {expected}, got {got}")]
    InvalidDimension {
        expected: usize,
        got: usize,
    },
    #[error("Invalid bounds: min values must be less than max values")]
    InvalidBounds,
    #[error("Optimization failed: {0}")]
    OptimizationError(String),
}

/// Result type for metaheuristic operations
pub type Result<T> = std::result::Result<T, MetaheuristicError>;
