use ndarray::{Array1, Array2};
use rand::Rng;
use crate::{MetaheuristicError, Result};
use super::{ObjectiveFunction, OptimizationParams, OptimizationResult, argmin};

/// Parameters specific to the Artificial Bee Colony Optimization algorithm
#[derive(Debug, Clone)]
pub struct ABCOParams {
    /// General optimization parameters
    pub opt_params: OptimizationParams,
    /// Maximum number of trials before abandoning a food source
    pub limit: usize,
}

impl Default for ABCOParams {
    fn default() -> Self {
        Self {
            opt_params: OptimizationParams::default(),
            limit: 100,
        }
    }
}

/// Implementation of the Artificial Bee Colony Optimization algorithm
pub struct ABCO<'a> {
    objective: &'a dyn ObjectiveFunction,
    params: ABCOParams,
}

impl<'a> ABCO<'a> {
    /// Create a new instance of ABCO
    pub fn new(objective: &'a dyn ObjectiveFunction, params: ABCOParams) -> Self {
        Self { objective, params }
    }
    
    /// Run the optimization algorithm
    pub fn optimize(&self) -> Result<OptimizationResult> {
        let mut rng = rand::thread_rng();
        
        // Initialize population
        let mut population = self.initialize_population(&mut rng)?;
        let mut fitness = self.evaluate_population(&population);
        let mut trials = Array1::zeros(self.params.opt_params.population_size);
        
        let best_idx = argmin(&fitness).expect("Failed to find minimum fitness");
        let mut best_solution = population.row(best_idx).to_owned();
        let mut best_fitness = fitness[best_idx];
        
        let mut evaluations = self.params.opt_params.population_size;
        
        for _iteration in 0..self.params.opt_params.max_iterations {
            // Employed Bee Phase
            self.employed_bee_phase(&mut population, &mut fitness, &mut trials, &mut rng)?;
            evaluations += self.params.opt_params.population_size;
            
            // Onlooker Bee Phase
            self.onlooker_bee_phase(&mut population, &mut fitness, &mut trials, &mut rng)?;
            evaluations += self.params.opt_params.population_size;
            
            // Scout Bee Phase
            self.scout_bee_phase(&mut population, &mut fitness, &mut trials, &mut rng)?;
            
            // Update best solution
            if let Some(min_idx) = argmin(&fitness) {
                if fitness[min_idx] < best_fitness {
                    best_fitness = fitness[min_idx];
                    best_solution = population.row(min_idx).to_owned();
                }
            }
            
            // Check termination criteria
            if let Some(target) = self.params.opt_params.target_value {
                if best_fitness <= target {
                    break;
                }
            }
        }
        
        Ok(OptimizationResult {
            best_solution,
            best_fitness,
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
    
    /// Employed Bee Phase
    fn employed_bee_phase(
        &self,
        population: &mut Array2<f64>,
        fitness: &mut Array1<f64>,
        trials: &mut Array1<f64>,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        let dims = self.objective.dimensions();
        
        for i in 0..self.params.opt_params.population_size {
            // Generate random partner
            let partner = rng.gen_range(0..self.params.opt_params.population_size);
            let dim = rng.gen_range(0..dims);
            
            // Create new solution
            let mut new_solution = population.row(i).to_owned();
            let phi = rng.gen_range(-1.0..=1.0);
            new_solution[dim] = population[[i, dim]] + phi * (population[[i, dim]] - population[[partner, dim]]);
            
            // Ensure bounds
            new_solution[dim] = new_solution[dim].clamp(lower_bounds[dim], upper_bounds[dim]);
            
            // Evaluate new solution
            let new_fitness = self.objective.evaluate(&new_solution);
            
            // Update if better
            if new_fitness < fitness[i] {
                population.row_mut(i).assign(&new_solution);
                fitness[i] = new_fitness;
                trials[i] = 0.0;
            } else {
                trials[i] += 1.0;
            }
        }
        
        Ok(())
    }
    
    /// Onlooker Bee Phase
    fn onlooker_bee_phase(
        &self,
        population: &mut Array2<f64>,
        fitness: &mut Array1<f64>,
        trials: &mut Array1<f64>,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        let dims = self.objective.dimensions();
        
        // Calculate selection probabilities
        let max_fitness = fitness.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_fitness = fitness.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max_fitness - min_fitness;
        
        let mut probabilities = Array1::zeros(self.params.opt_params.population_size);
        for i in 0..self.params.opt_params.population_size {
            probabilities[i] = if range > 0.0 {
                (max_fitness - fitness[i]) / range
            } else {
                1.0 / self.params.opt_params.population_size as f64
            };
        }
        
        // Onlooker bee phase
        for _ in 0..self.params.opt_params.population_size {
            // Select food source
            let mut cumsum = 0.0;
            let r = rng.gen::<f64>();
            let mut selected = 0;
            
            for (i, &p) in probabilities.iter().enumerate() {
                cumsum += p;
                if cumsum > r {
                    selected = i;
                    break;
                }
            }
            
            // Generate new solution
            let partner = rng.gen_range(0..self.params.opt_params.population_size);
            let dim = rng.gen_range(0..dims);
            
            let mut new_solution = population.row(selected).to_owned();
            let phi = rng.gen_range(-1.0..=1.0);
            new_solution[dim] = population[[selected, dim]] + phi * (population[[selected, dim]] - population[[partner, dim]]);
            
            // Ensure bounds
            new_solution[dim] = new_solution[dim].clamp(lower_bounds[dim], upper_bounds[dim]);
            
            // Evaluate new solution
            let new_fitness = self.objective.evaluate(&new_solution);
            
            // Update if better
            if new_fitness < fitness[selected] {
                population.row_mut(selected).assign(&new_solution);
                fitness[selected] = new_fitness;
                trials[selected] = 0.0;
            } else {
                trials[selected] += 1.0;
            }
        }
        
        Ok(())
    }
    
    /// Scout Bee Phase
    fn scout_bee_phase(
        &self,
        population: &mut Array2<f64>,
        fitness: &mut Array1<f64>,
        trials: &mut Array1<f64>,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        let dims = self.objective.dimensions();
        
        for i in 0..self.params.opt_params.population_size {
            if trials[i] > self.params.limit as f64 {
                // Abandon and generate new solution
                for j in 0..dims {
                    population[[i, j]] = rng.gen_range(lower_bounds[j]..=upper_bounds[j]);
                }
                fitness[i] = self.objective.evaluate(&population.row(i).to_owned());
                trials[i] = 0.0;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    /// Simple sphere function for testing
    struct SphereProblem {
        dims: usize,
    }
    
    impl SphereProblem {
        fn new(dims: usize) -> Self {
            Self { dims }
        }
    }
    
    impl ObjectiveFunction for SphereProblem {
        fn evaluate(&self, x: &Array1<f64>) -> f64 {
            x.iter().map(|&x| x * x).sum()
        }
        
        fn dimensions(&self) -> usize {
            self.dims
        }
        
        fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
            (
                vec![-5.0; self.dims],
                vec![5.0; self.dims]
            )
        }
    }
    
    #[test]
    fn test_abco_optimization() {
        let problem = SphereProblem::new(2);
        let params = ABCOParams::default();
        let optimizer = ABCO::new(&problem, params);
        
        let result = optimizer.optimize().unwrap();
        
        // The global minimum of the sphere function is at (0,0) with f(x)=0
        assert!(result.best_fitness < 1e-2); // Allow for some numerical error
    }
}
