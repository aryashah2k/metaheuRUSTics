use ndarray::Array1;

pub mod abco;
pub mod de;
pub mod ga;
pub mod gwo;
pub mod pso;
pub mod sa;

// Re-export common types and algorithms
pub use self::abco::{ABCO, ABCOParams};
pub use self::de::{DE, DEParams, DEStrategy};
pub use self::ga::{GA, GAParams};
pub use self::gwo::{GWO, GWOParams};
pub use self::pso::{PSO, PSOParams};
pub use self::sa::{SA, SAParams};

/// Trait for objective functions to be optimized
pub trait ObjectiveFunction {
    /// Evaluate the objective function at a given point
    fn evaluate(&self, x: &Array1<f64>) -> f64;
    
    /// Get the number of dimensions of the problem
    fn dimensions(&self) -> usize;
    
    /// Get the bounds of the search space
    fn bounds(&self) -> (Vec<f64>, Vec<f64>);
}

/// Parameters for optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizationParams {
    /// Size of the population
    pub population_size: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Target value for early stopping
    pub target_value: Option<f64>,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_iterations: 1000,
            target_value: None,
        }
    }
}

/// Result of an optimization run
#[derive(Debug)]
pub struct OptimizationResult {
    /// Best solution found
    pub best_solution: Array1<f64>,
    /// Best fitness value found
    pub best_fitness: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of function evaluations
    pub evaluations: usize,
}

/// Helper function to find the index of the minimum value in an array
pub(crate) fn argmin(array: &Array1<f64>) -> Option<usize> {
    array.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
}
