use ndarray::{Array1, Array2};
use rand::Rng;
use crate::{MetaheuristicError, Result};
use super::{ObjectiveFunction, OptimizationParams, OptimizationResult, argmin};

/// Parameters specific to the Genetic Algorithm
#[derive(Debug, Clone)]
pub struct GAParams {
    /// General optimization parameters
    pub opt_params: OptimizationParams,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Mutation probability
    pub mutation_rate: f64,
    /// Tournament size for selection
    pub tournament_size: usize,
}

impl Default for GAParams {
    fn default() -> Self {
        Self {
            opt_params: OptimizationParams::default(),
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            tournament_size: 3,
        }
    }
}

/// Implementation of the Genetic Algorithm
pub struct GA<'a> {
    objective: &'a dyn ObjectiveFunction,
    params: GAParams,
}

impl<'a> GA<'a> {
    /// Create a new instance of GA
    pub fn new(objective: &'a dyn ObjectiveFunction, params: GAParams) -> Self {
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
            let mut new_population = Array2::zeros(population.raw_dim());
            
            // Elitism: keep the best solution
            new_population.row_mut(0).assign(&best_solution);
            
            // Generate new solutions
            for i in 1..self.params.opt_params.population_size {
                let parent1 = self.tournament_selection(&fitness, &mut rng);
                let parent2 = self.tournament_selection(&fitness, &mut rng);
                
                let child = self.crossover(
                    population.row(parent1).to_owned(),
                    population.row(parent2).to_owned(),
                    &mut rng,
                );
                
                let mutated_child = self.mutate(child, &mut rng);
                new_population.row_mut(i).assign(&mutated_child);
            }
            
            population = new_population;
            fitness = self.evaluate_population(&population);
            evaluations += self.params.opt_params.population_size;
            
            // Update best solution
            if let Some(idx) = argmin(&fitness) {
                if fitness[idx] < best_fitness {
                    best_solution = population.row(idx).to_owned();
                    best_fitness = fitness[idx];
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
    
    /// Initialize a random population
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
    
    /// Evaluate the fitness of each individual in the population
    fn evaluate_population(&self, population: &Array2<f64>) -> Array1<f64> {
        let mut fitness = Array1::zeros(population.nrows());
        for (i, individual) in population.rows().into_iter().enumerate() {
            fitness[i] = self.objective.evaluate(&individual.to_owned());
        }
        fitness
    }
    
    /// Tournament selection
    fn tournament_selection(&self, fitness: &Array1<f64>, rng: &mut impl Rng) -> usize {
        let mut best_idx = rng.gen_range(0..self.params.opt_params.population_size);
        let mut best_fitness = fitness[best_idx];
        
        for _ in 1..self.params.tournament_size {
            let idx = rng.gen_range(0..self.params.opt_params.population_size);
            if fitness[idx] < best_fitness {
                best_idx = idx;
                best_fitness = fitness[idx];
            }
        }
        
        best_idx
    }
    
    /// Perform crossover between two parents
    fn crossover(
        &self,
        parent1: Array1<f64>,
        parent2: Array1<f64>,
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        let mut child = parent1.clone();
        
        let crossover_point = rng.gen_range(0..parent1.len());
        for i in crossover_point..parent1.len() {
            child[i] = parent2[i];
        }
        
        child
    }
    
    /// Perform mutation on an individual
    fn mutate(&self, individual: Array1<f64>, rng: &mut impl Rng) -> Array1<f64> {
        let (lower_bounds, upper_bounds) = self.objective.bounds();
        let mutation_strength = 0.1; // 10% of the range
        
        let mut mutated_individual = individual.clone();
        for i in 0..individual.len() {
            let range = (upper_bounds[i] - lower_bounds[i]) * mutation_strength;
            let delta = (rng.gen::<f64>() - 0.5) * 2.0 * range;
            mutated_individual[i] = (mutated_individual[i] + delta).clamp(lower_bounds[i], upper_bounds[i]);
        }
        
        mutated_individual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_function::{Sphere, TestFunction};
    
    #[test]
    fn test_ga_optimization() {
        let problem = Sphere::new(2);
        let params = GAParams::default();
        let optimizer = GA::new(&problem, params);
        
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
    fn test_ga_bounds() {
        let problem = Sphere::new(2);
        let params = GAParams {
            opt_params: OptimizationParams {
                population_size: 10,
                max_iterations: 50,
                ..Default::default()
            },
            ..Default::default()
        };
        let optimizer = GA::new(&problem, params);
        
        let result = optimizer.optimize().unwrap();
        
        // Check if solution is within bounds
        let (lower_bounds, upper_bounds) = problem.bounds();
        for (i, &x) in result.best_solution.iter().enumerate() {
            assert!(x >= lower_bounds[i] && x <= upper_bounds[i]);
        }
    }
}
