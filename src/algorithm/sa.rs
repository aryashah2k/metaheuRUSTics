use ndarray::Array1;
use rand::Rng;
use crate::{MetaheuristicError, Result};
use super::{ObjectiveFunction, OptimizationParams, OptimizationResult};

/// Parameters specific to the Simulated Annealing algorithm
#[derive(Debug, Clone)]
pub struct SAParams {
    /// General optimization parameters
    pub opt_params: OptimizationParams,
    /// Initial temperature
    pub initial_temp: f64,
    /// Cooling rate (alpha)
    pub cooling_rate: f64,
    /// Number of iterations at each temperature
    pub iterations_per_temp: usize,
    /// Minimum temperature
    pub min_temp: f64,
}

impl Default for SAParams {
    fn default() -> Self {
        Self {
            opt_params: OptimizationParams::default(),
            initial_temp: 100.0,
            cooling_rate: 0.95,
            iterations_per_temp: 50,
            min_temp: 1e-10,
        }
    }
}

/// Implementation of the Simulated Annealing algorithm
pub struct SA<'a> {
    objective: &'a dyn ObjectiveFunction,
    params: SAParams,
}

impl<'a> SA<'a> {
    /// Create a new instance of SA
    pub fn new(objective: &'a dyn ObjectiveFunction, params: SAParams) -> Self {
        Self { objective, params }
    }
    
    /// Run the optimization algorithm
    pub fn optimize(&self) -> Result<OptimizationResult> {
        let mut rng = rand::thread_rng();
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        let dims = self.objective.dimensions();
        
        // Initialize current solution
        let mut current_solution = self.initialize_solution(&mut rng)?;
        let mut current_energy = self.objective.evaluate(&current_solution);
        
        // Initialize best solution
        let mut best_solution = current_solution.clone();
        let mut best_energy = current_energy;
        
        let mut temp = self.params.initial_temp;
        let mut iteration = 0;
        let mut evaluations = 1;
        
        while temp > self.params.min_temp && iteration < self.params.opt_params.max_iterations {
            for _ in 0..self.params.iterations_per_temp {
                // Generate neighbor solution
                let mut neighbor = current_solution.clone();
                self.perturb_solution(&mut neighbor, temp, &mut rng);
                
                // Ensure bounds
                for i in 0..dims {
                    neighbor[i] = neighbor[i].clamp(lower_bounds[i], upper_bounds[i]);
                }
                
                // Evaluate neighbor
                let neighbor_energy = self.objective.evaluate(&neighbor);
                evaluations += 1;
                
                // Calculate energy difference
                let delta_e = neighbor_energy - current_energy;
                
                // Accept or reject new solution
                if self.accept_solution(delta_e, temp, &mut rng) {
                    current_solution = neighbor;
                    current_energy = neighbor_energy;
                    
                    // Update best solution
                    if current_energy < best_energy {
                        best_solution = current_solution.clone();
                        best_energy = current_energy;
                    }
                }
            }
            
            // Cool down
            temp *= self.params.cooling_rate;
            iteration += 1;
            
            // Check termination criteria
            if let Some(target) = self.params.opt_params.target_value {
                if best_energy <= target {
                    break;
                }
            }
        }
        
        Ok(OptimizationResult {
            best_solution,
            best_fitness: best_energy,
            iterations: iteration,
            evaluations,
        })
    }
    
    /// Initialize a random solution
    fn initialize_solution(&self, rng: &mut impl Rng) -> Result<Array1<f64>> {
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        let dims = self.objective.dimensions();
        
        if lower_bounds.len() != dims || upper_bounds.len() != dims {
            return Err(MetaheuristicError::InvalidDimension {
                expected: dims,
                got: lower_bounds.len(),
            });
        }
        
        let mut solution = Array1::zeros(dims);
        for i in 0..dims {
            solution[i] = rng.gen_range(lower_bounds[i]..=upper_bounds[i]);
        }
        
        Ok(solution)
    }
    
    /// Perturb the current solution
    fn perturb_solution(&self, solution: &mut Array1<f64>, temp: f64, rng: &mut impl Rng) {
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        
        for i in 0..solution.len() {
            // Scale perturbation with temperature
            let range = (upper_bounds[i] - lower_bounds[i]) * temp / self.params.initial_temp;
            let delta = (rng.gen::<f64>() - 0.5) * 2.0 * range;
            solution[i] += delta;
        }
    }
    
    /// Decide whether to accept a solution
    fn accept_solution(&self, delta_e: f64, temp: f64, rng: &mut impl Rng) -> bool {
        if delta_e <= 0.0 {
            true
        } else {
            let probability = (-delta_e / temp).exp();
            rng.gen::<f64>() < probability
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_function::{Sphere, TestFunction};
    
    #[test]
    fn test_sa_optimization() {
        let problem = Sphere::new(2);
        let params = SAParams::default();
        let optimizer = SA::new(&problem, params);
        
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
    fn test_sa_bounds() {
        let problem = Sphere::new(2);
        let params = SAParams {
            opt_params: OptimizationParams {
                max_iterations: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        let optimizer = SA::new(&problem, params);
        
        let result = optimizer.optimize().unwrap();
        
        // Check if solution is within bounds
        let (lower_bounds, upper_bounds) = problem.bounds();
        for (i, &x) in result.best_solution.iter().enumerate() {
            assert!(x >= lower_bounds[i] && x <= upper_bounds[i]);
        }
    }
    
    #[test]
    fn test_sa_temperature_decay() {
        let problem = Sphere::new(2);
        let params = SAParams {
            initial_temp: 100.0,
            cooling_rate: 0.8,
            min_temp: 1.0,
            ..Default::default()
        };
        let optimizer = SA::new(&problem, params);
        
        let result = optimizer.optimize().unwrap();
        
        // Verify that optimization completed
        assert!(result.iterations > 0);
        assert!(result.best_fitness < 1.0);
    }
}
