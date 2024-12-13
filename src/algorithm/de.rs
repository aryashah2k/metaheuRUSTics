use ndarray::{Array1, Array2};
use rand::Rng;
use rand::seq::SliceRandom;
use crate::{MetaheuristicError, Result};
use super::{ObjectiveFunction, OptimizationParams, OptimizationResult, argmin};

/// Different DE mutation strategies
#[derive(Debug, Clone, Copy)]
pub enum DEStrategy {
    RandOneBin,
    BestOneBin,
    RandTwoBin,
}

/// Parameters specific to the Differential Evolution algorithm
#[derive(Debug, Clone)]
pub struct DEParams {
    /// General optimization parameters
    pub opt_params: OptimizationParams,
    /// Mutation strategy
    pub strategy: DEStrategy,
    /// Crossover rate
    pub crossover_rate: f64,
    /// Mutation factor (F)
    pub mutation_factor: f64,
}

impl Default for DEParams {
    fn default() -> Self {
        Self {
            opt_params: OptimizationParams::default(),
            strategy: DEStrategy::RandOneBin,
            crossover_rate: 0.9,
            mutation_factor: 0.8,
        }
    }
}

/// Implementation of the Differential Evolution algorithm
pub struct DE<'a> {
    objective: &'a dyn ObjectiveFunction,
    params: DEParams,
}

impl<'a> DE<'a> {
    /// Create a new instance of DE
    pub fn new(objective: &'a dyn ObjectiveFunction, params: DEParams) -> Self {
        Self { objective, params }
    }
    
    /// Run the optimization algorithm
    pub fn optimize(&self) -> Result<OptimizationResult> {
        let mut rng = rand::thread_rng();
        
        // Initialize population
        let mut population = self.initialize_population(&mut rng)?;
        let mut fitness = self.evaluate_population(&population);
        
        let best_idx = argmin(&fitness).expect("Failed to find minimum fitness");
        let mut best_solution = population.row(best_idx).to_owned();
        let mut best_fitness = fitness[best_idx];
        
        let mut evaluations = self.params.opt_params.population_size;
        
        for _iteration in 0..self.params.opt_params.max_iterations {
            // Create new population through mutation and crossover
            let mut new_population = Array2::zeros(population.raw_dim());
            
            for i in 0..self.params.opt_params.population_size {
                let trial_vector = match self.params.strategy {
                    DEStrategy::RandOneBin => self.rand_1_bin(i, &population, &mut rng),
                    DEStrategy::BestOneBin => self.best_1_bin(i, &population, best_idx, &mut rng),
                    DEStrategy::RandTwoBin => self.rand_2_bin(i, &population, &mut rng),
                };
                
                // Evaluate trial vector
                let trial_fitness = self.objective.evaluate(&trial_vector);
                evaluations += 1;
                
                // Selection
                if trial_fitness <= fitness[i] {
                    new_population.row_mut(i).assign(&trial_vector);
                    fitness[i] = trial_fitness;
                    
                    // Update best solution
                    if trial_fitness < best_fitness {
                        best_fitness = trial_fitness;
                        best_solution = trial_vector.clone();
                    }
                } else {
                    new_population.row_mut(i).assign(&population.row(i));
                }
            }
            
            population = new_population;
            
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
    
    /// DE/rand/1/bin strategy
    fn rand_1_bin(&self, target_idx: usize, population: &Array2<f64>, rng: &mut impl Rng) -> Array1<f64> {
        let dims = self.objective.dimensions();
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        
        // Select random indices different from target
        let mut indices: Vec<usize> = (0..self.params.opt_params.population_size)
            .filter(|&x| x != target_idx)
            .collect();
        indices.shuffle(rng);
        let (r1, r2, r3) = (indices[0], indices[1], indices[2]);
        
        // Create trial vector
        let mut trial = population.row(target_idx).to_owned();
        let j_rand = rng.gen_range(0..dims);
        
        for j in 0..dims {
            if j == j_rand || rng.gen::<f64>() < self.params.crossover_rate {
                trial[j] = population[[r1, j]] + self.params.mutation_factor * (population[[r2, j]] - population[[r3, j]]);
                trial[j] = trial[j].clamp(lower_bounds[j], upper_bounds[j]);
            }
        }
        
        trial
    }
    
    /// DE/best/1/bin strategy
    fn best_1_bin(&self, target_idx: usize, population: &Array2<f64>, best_idx: usize, rng: &mut impl Rng) -> Array1<f64> {
        let dims = self.objective.dimensions();
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        
        // Select random indices different from target and best
        let mut indices: Vec<usize> = (0..self.params.opt_params.population_size)
            .filter(|&x| x != target_idx && x != best_idx)
            .collect();
        indices.shuffle(rng);
        let (r1, r2) = (indices[0], indices[1]);
        
        // Create trial vector
        let mut trial = population.row(target_idx).to_owned();
        let j_rand = rng.gen_range(0..dims);
        
        for j in 0..dims {
            if j == j_rand || rng.gen::<f64>() < self.params.crossover_rate {
                trial[j] = population[[best_idx, j]] + self.params.mutation_factor * (population[[r1, j]] - population[[r2, j]]);
                trial[j] = trial[j].clamp(lower_bounds[j], upper_bounds[j]);
            }
        }
        
        trial
    }
    
    /// DE/rand/2/bin strategy
    fn rand_2_bin(&self, target_idx: usize, population: &Array2<f64>, rng: &mut impl Rng) -> Array1<f64> {
        let dims = self.objective.dimensions();
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        
        // Select random indices different from target
        let mut indices: Vec<usize> = (0..self.params.opt_params.population_size)
            .filter(|&x| x != target_idx)
            .collect();
        indices.shuffle(rng);
        let (r1, r2, r3, r4, r5) = (indices[0], indices[1], indices[2], indices[3], indices[4]);
        
        // Create trial vector
        let mut trial = population.row(target_idx).to_owned();
        let j_rand = rng.gen_range(0..dims);
        
        for j in 0..dims {
            if j == j_rand || rng.gen::<f64>() < self.params.crossover_rate {
                trial[j] = population[[r1, j]] + 
                    self.params.mutation_factor * (population[[r2, j]] - population[[r3, j]]) +
                    self.params.mutation_factor * (population[[r4, j]] - population[[r5, j]]);
                trial[j] = trial[j].clamp(lower_bounds[j], upper_bounds[j]);
            }
        }
        
        trial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_function::{Sphere, TestFunction};
    
    #[test]
    fn test_de_optimization() {
        let problem = Sphere::new(2);
        let params = DEParams::default();
        let optimizer = DE::new(&problem, params);
        
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
    fn test_de_strategies() {
        let problem = Sphere::new(2);
        
        // Test all strategies
        let strategies = [
            DEStrategy::RandOneBin,
            DEStrategy::BestOneBin,
            DEStrategy::RandTwoBin,
        ];
        
        for strategy in strategies.iter() {
            let params = DEParams {
                strategy: *strategy,
                ..DEParams::default()
            };
            let optimizer = DE::new(&problem, params);
            let result = optimizer.optimize().unwrap();
            
            // All strategies should be able to find the minimum
            assert!(result.best_fitness < 1e-2);
        }
    }
}
