use ndarray::{Array1, Array2};
use rand::Rng;
use crate::{MetaheuristicError, Result};
use super::{ObjectiveFunction, OptimizationParams, OptimizationResult};

/// Parameters specific to the Grey Wolf Optimizer
#[derive(Debug, Clone)]
pub struct GWOParams {
    /// General optimization parameters
    pub opt_params: OptimizationParams,
    /// Parameter a decay rate
    pub a_decay: f64,
}

impl Default for GWOParams {
    fn default() -> Self {
        Self {
            opt_params: OptimizationParams::default(),
            a_decay: 2.0,
        }
    }
}

/// Implementation of the Grey Wolf Optimizer
pub struct GWO<'a> {
    objective: &'a dyn ObjectiveFunction,
    params: GWOParams,
}

impl<'a> GWO<'a> {
    /// Create a new instance of GWO
    pub fn new(objective: &'a dyn ObjectiveFunction, params: GWOParams) -> Self {
        Self { objective, params }
    }
    
    /// Run the optimization algorithm
    pub fn optimize(&self) -> Result<OptimizationResult> {
        let mut rng = rand::thread_rng();
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        let dims = self.objective.dimensions();
        
        // Initialize population
        let mut population = self.initialize_population(&mut rng)?;
        let mut fitness = self.evaluate_population(&population);
        
        // Initialize alpha, beta, and delta wolves
        let mut sorted_indices = (0..fitness.len()).collect::<Vec<_>>();
        sorted_indices.sort_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap());
        
        let mut alpha = population.row(sorted_indices[0]).to_owned();
        let mut beta = population.row(sorted_indices[1]).to_owned();
        let mut delta = population.row(sorted_indices[2]).to_owned();
        let mut alpha_score = fitness[sorted_indices[0]];
        let mut beta_score = fitness[sorted_indices[1]];
        let mut delta_score = fitness[sorted_indices[2]];
        
        for iteration in 0..self.params.opt_params.max_iterations {
            // Update a
            let a = self.params.a_decay * (1.0 - iteration as f64 / self.params.opt_params.max_iterations as f64);
            
            // Update each wolf's position
            for i in 0..self.params.opt_params.population_size {
                let mut new_position = Array1::zeros(dims);
                
                // Calculate new position based on alpha, beta, and delta
                for j in 0..dims {
                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    
                    // Position updating coefficients
                    let c1 = 2.0 * r1;
                    let c2 = 2.0 * r2;
                    let a1 = 2.0 * a * r1 - a;
                    let a2 = 2.0 * a * r2 - a;
                    
                    // Calculate positions based on alpha, beta, and delta
                    let x1 = alpha[j] - a1 * (c1 * alpha[j] - population[[i, j]]).abs();
                    let x2 = beta[j] - a2 * (c2 * beta[j] - population[[i, j]]).abs();
                    let x3 = delta[j] - a1 * (c1 * delta[j] - population[[i, j]]).abs();
                    
                    // Average position
                    new_position[j] = (x1 + x2 + x3) / 3.0;
                    
                    // Bound position
                    new_position[j] = new_position[j].clamp(lower_bounds[j], upper_bounds[j]);
                }
                
                // Update position
                population.row_mut(i).assign(&new_position);
            }
            
            // Evaluate new positions
            fitness = self.evaluate_population(&population);
            
            // Update alpha, beta, and delta
            sorted_indices.sort_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap());
            
            if fitness[sorted_indices[0]] < alpha_score {
                alpha = population.row(sorted_indices[0]).to_owned();
                alpha_score = fitness[sorted_indices[0]];
            }
            
            if fitness[sorted_indices[1]] < beta_score {
                beta = population.row(sorted_indices[1]).to_owned();
                beta_score = fitness[sorted_indices[1]];
            }
            
            if fitness[sorted_indices[2]] < delta_score {
                delta = population.row(sorted_indices[2]).to_owned();
                delta_score = fitness[sorted_indices[2]];
            }
            
            // Check termination criteria
            if let Some(target) = self.params.opt_params.target_value {
                if alpha_score <= target {
                    break;
                }
            }
        }
        
        Ok(OptimizationResult {
            best_solution: alpha,
            best_fitness: alpha_score,
            iterations: self.params.opt_params.max_iterations,
            evaluations: self.params.opt_params.max_iterations * self.params.opt_params.population_size,
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
    fn test_gwo_optimization() {
        let problem = Sphere::new(2);
        let params = GWOParams::default();
        let optimizer = GWO::new(&problem, params);
        
        let result = optimizer.optimize().unwrap();
        
        // The global minimum of the sphere function is at (0,0) with f(x)=0
        assert!(result.best_fitness < 1e-2); // Allow for some numerical error
        
        // Check if the solution is close to the known global minimum
        let global_min = problem.global_minimum_position();
        for (x, x_min) in result.best_solution.iter().zip(global_min.iter()) {
            assert!((x - x_min).abs() < 1e-1);
        }
    }
    
    #[test]
    fn test_gwo_bounds() {
        let problem = Sphere::new(2);
        let params = GWOParams {
            opt_params: OptimizationParams {
                population_size: 10,
                max_iterations: 50,
                ..Default::default()
            },
            ..Default::default()
        };
        let optimizer = GWO::new(&problem, params);
        
        let result = optimizer.optimize().unwrap();
        
        // Check if solution is within bounds
        let (lower_bounds, upper_bounds) = problem.bounds();
        for (i, &x) in result.best_solution.iter().enumerate() {
            assert!(x >= lower_bounds[i] && x <= upper_bounds[i]);
        }
    }
}
