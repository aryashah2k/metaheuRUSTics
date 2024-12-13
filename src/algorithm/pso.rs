use ndarray::{Array1, Array2};
use rand::Rng;
use crate::{MetaheuristicError, Result};
use super::{ObjectiveFunction, OptimizationParams, OptimizationResult, argmin};

/// Parameters specific to the Particle Swarm Optimization algorithm
#[derive(Debug, Clone)]
pub struct PSOParams {
    /// General optimization parameters
    pub opt_params: OptimizationParams,
    /// Inertia weight
    pub w: f64,
    /// Cognitive learning factor
    pub c1: f64,
    /// Social learning factor
    pub c2: f64,
    /// Inertia weight decay rate
    pub w_decay: f64,
}

impl Default for PSOParams {
    fn default() -> Self {
        Self {
            opt_params: OptimizationParams::default(),
            w: 0.9,
            c1: 2.0,
            c2: 2.0,
            w_decay: 0.0,
        }
    }
}

/// Implementation of the Particle Swarm Optimization algorithm
pub struct PSO<'a> {
    objective: &'a dyn ObjectiveFunction,
    params: PSOParams,
}

impl<'a> PSO<'a> {
    /// Create a new instance of PSO
    pub fn new(objective: &'a dyn ObjectiveFunction, params: PSOParams) -> Self {
        Self { objective, params }
    }
    
    /// Run the optimization algorithm
    pub fn optimize(&self) -> Result<OptimizationResult> {
        let mut rng = rand::thread_rng();
        
        // Initialize population and velocities
        let mut positions = self.initialize_population(&mut rng)?;
        let mut velocities: Array2<f64> = Array2::zeros((self.params.opt_params.population_size, self.objective.dimensions()));
        
        // Initialize velocities with random values in [-1, 1]
        for i in 0..self.params.opt_params.population_size {
            for j in 0..self.objective.dimensions() {
                velocities[[i, j]] = rng.gen_range(-1.0..=1.0);
            }
        }
        
        // Initialize personal best
        let mut personal_best = positions.clone();
        let mut personal_best_values = self.evaluate_population(&positions);
        
        // Initialize global best
        let best_idx = argmin(&personal_best_values).expect("Failed to find minimum fitness");
        let mut global_best_position = positions.row(best_idx).to_owned();
        let mut global_best_value = personal_best_values[best_idx];
        
        let mut evaluations = self.params.opt_params.population_size;
        
        for _iteration in 0..self.params.opt_params.max_iterations {
            // Update velocities and positions
            for i in 0..self.params.opt_params.population_size {
                for j in 0..self.objective.dimensions() {
                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    
                    // Update velocity
                    velocities[[i, j]] = self.params.w * velocities[[i, j]]
                        + self.params.c1 * r1 * (personal_best[[i, j]] - positions[[i, j]])
                        + self.params.c2 * r2 * (global_best_position[j] - positions[[i, j]]);
                    
                    // Update position
                    positions[[i, j]] += velocities[[i, j]];
                    
                    // Clamp position to bounds
                    let (lower_bounds, upper_bounds) = self.objective.bounds();
                    positions[[i, j]] = positions[[i, j]].clamp(lower_bounds[j], upper_bounds[j]);
                }
            }
            
            // Evaluate new positions
            let current_values = self.evaluate_population(&positions);
            evaluations += self.params.opt_params.population_size;
            
            // Update personal and global best
            for i in 0..self.params.opt_params.population_size {
                if current_values[i] < personal_best_values[i] {
                    personal_best_values[i] = current_values[i];
                    personal_best.row_mut(i).assign(&positions.row(i));
                    
                    if current_values[i] < global_best_value {
                        global_best_value = current_values[i];
                        global_best_position = positions.row(i).to_owned();
                    }
                }
            }
            
            // Check termination criteria
            if let Some(target) = self.params.opt_params.target_value {
                if global_best_value <= target {
                    break;
                }
            }
        }
        
        Ok(OptimizationResult {
            best_solution: global_best_position,
            best_fitness: global_best_value,
            iterations: self.params.opt_params.max_iterations,
            evaluations,
        })
    }
    
    /// Initialize the population
    fn initialize_population(&self, rng: &mut impl Rng) -> Result<Array2<f64>> {
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        let dims = self.objective.dimensions();
        
        if lower_bounds.len() != dims || upper_bounds.len() != dims {
            return Err(MetaheuristicError::InvalidDimension {
                expected: dims,
                got: lower_bounds.len(),
            });
        }
        
        let mut population = Array2::zeros((self.params.opt_params.population_size, dims));
        for i in 0..self.params.opt_params.population_size {
            for j in 0..dims {
                population[[i, j]] = rng.gen_range(lower_bounds[j]..=upper_bounds[j]);
            }
        }
        
        Ok(population)
    }
    
    /// Evaluate the entire population
    fn evaluate_population(&self, population: &Array2<f64>) -> Array1<f64> {
        let mut fitness = Array1::zeros(population.nrows());
        for (i, row) in population.rows().into_iter().enumerate() {
            fitness[i] = self.objective.evaluate(&row.to_owned());
        }
        fitness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_function::{Sphere, TestFunction};
    
    #[test]
    fn test_pso_optimization() {
        let problem = Sphere::new(2);
        let params = PSOParams::default();
        let optimizer = PSO::new(&problem, params);
        
        let result = optimizer.optimize().unwrap();
        
        // The global minimum of the sphere function is at (0,0) with f(x)=0
        assert!(result.best_fitness < 1e-2); // Allow for some numerical error
        
        // Check if the solution is close to the known global minimum
        let global_min = problem.global_minimum_position();
        for (x, x_min) in result.best_solution.iter().zip(global_min.iter()) {
            assert!((x - x_min).abs() < 1e-1);
        }
    }
}
