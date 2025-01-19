use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use crate::algorithm::{ObjectiveFunction, OptimizationParams, OptimizationResult};

/// Parameters for the Adaptive Chaotic Grey Wolf Optimizer
#[derive(Debug, Clone)]
pub struct ACGWOParams {
    /// Common optimization parameters
    pub opt_params: OptimizationParams,
    /// Initial value of lambda for chaotic map
    pub lambda: f64,
    /// Adaptive parameter for a
    pub a_decay: f64,
}

impl Default for ACGWOParams {
    fn default() -> Self {
        Self {
            opt_params: OptimizationParams::default(),
            lambda: 0.5,
            a_decay: 0.01,
        }
    }
}

/// Adaptive Chaotic Grey Wolf Optimizer implementation
pub struct ACGWO<'a, F: ObjectiveFunction + Sized> {
    objective_fn: &'a F,
    params: ACGWOParams,
}

impl<'a, F: ObjectiveFunction + Sized> ACGWO<'a, F> {
    /// Create a new instance of ACGWO
    pub fn new(objective_fn: &'a F, params: ACGWOParams) -> Self {
        Self {
            objective_fn,
            params,
        }
    }

    /// Chaotic map function
    fn chaotic_map(&self, x: f64) -> f64 {
        // Using logistic map with improved chaos
        let r = 3.9; // Increased from default 3.7 to create more chaos
        r * x * (1.0 - x)
    }

    /// Initialize population using chaotic map
    fn initialize_population(&self, size: usize, dim: usize, bounds: &(Vec<f64>, Vec<f64>)) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut population = Array2::zeros((size, dim));
        
        for i in 0..size {
            for j in 0..dim {
                let random = rng.gen::<f64>();
                let chaotic = self.chaotic_map(random);
                population[[i, j]] = bounds.0[j] + chaotic * (bounds.1[j] - bounds.0[j]);
            }
        }
        
        population
    }

    /// Calculate the fitness of each wolf
    fn calculate_fitness(&self, population: &Array2<f64>) -> Vec<f64> {
        population.axis_iter(Axis(0))
            .map(|wolf| self.objective_fn.evaluate(&wolf.to_owned()))
            .collect()
    }

    /// Sort wolves based on fitness
    fn sort_wolves(&self, population: &Array2<f64>, fitness: &[f64]) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let mut indices: Vec<usize> = (0..fitness.len()).collect();
        indices.sort_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap());

        let alpha = population.row(indices[0]).to_owned();
        let beta = population.row(indices[1]).to_owned();
        let delta = population.row(indices[2]).to_owned();

        (alpha, beta, delta)
    }

    /// Update wolf positions
    fn update_positions(
        &self,
        population: &mut Array2<f64>,
        alpha: &Array1<f64>,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
        a: f64,
        bounds: &(Vec<f64>, Vec<f64>)
    ) {
        let (lower_bounds, upper_bounds) = bounds;
        let mut rng = rand::thread_rng();
        let pop_size = population.nrows();
        let dim = population.ncols();

        for i in 0..pop_size {
            for j in 0..dim {
                // Calculate A and C vectors
                let r1 = rng.gen::<f64>();
                let r2 = rng.gen::<f64>();
                let c = 2.0 * r2;

                // Use chaotic map for A
                let a_chaos = self.chaotic_map(a);
                let a1 = 2.0 * a_chaos * r1 - a_chaos;
                let a2 = 2.0 * a_chaos * r1 - a_chaos;
                let a3 = 2.0 * a_chaos * r1 - a_chaos;

                // Calculate distance to alpha, beta, and delta
                let d_alpha = (c * alpha[j] - population[[i, j]]).abs();
                let d_beta = (c * beta[j] - population[[i, j]]).abs();
                let d_delta = (c * delta[j] - population[[i, j]]).abs();

                // Calculate position updates with chaotic weights
                let x1 = alpha[j] - a1 * d_alpha;
                let x2 = beta[j] - a2 * d_beta;
                let x3 = delta[j] - a3 * d_delta;

                // Apply adaptive weights using lambda parameter
                let w1 = self.params.lambda / (1.0 + (-10.0 * d_alpha).exp());
                let w2 = (1.0 - self.params.lambda) / 2.0 / (1.0 + (-10.0 * d_beta).exp());
                let w3 = (1.0 - self.params.lambda) / 2.0 / (1.0 + (-10.0 * d_delta).exp());

                // Update position with weighted average
                let new_pos = (w1 * x1 + w2 * x2 + w3 * x3) / (w1 + w2 + w3);

                // Bound the position
                population[[i, j]] = new_pos.max(lower_bounds[j]).min(upper_bounds[j]);
            }
        }
    }

    /// Optimize the objective function
    pub fn optimize(&self) -> OptimizationResult {
        let dim = self.objective_fn.dimensions();
        let bounds = self.objective_fn.bounds();
        let pop_size = self.params.opt_params.population_size;
        let max_iter = self.params.opt_params.max_iterations;

        // Initialize population
        let mut population = self.initialize_population(pop_size, dim, &bounds);
        let mut fitness = self.calculate_fitness(&population);
        let (mut alpha, mut beta, mut delta) = self.sort_wolves(&population, &fitness);
        let mut alpha_score = self.objective_fn.evaluate(&alpha);

        let mut best_solution = alpha.clone();
        let mut best_fitness = alpha_score;
        let mut final_iter = 0;

        // Main loop
        for t in 0..max_iter {
            // Update a with decay
            let a = 2.0 * (1.0 - t as f64 / max_iter as f64) * (1.0 - self.params.a_decay).powi(t.try_into().unwrap());

            // Update positions
            self.update_positions(&mut population, &alpha, &beta, &delta, a, &bounds);

            // Evaluate new positions
            fitness = self.calculate_fitness(&population);
            let (new_alpha, new_beta, new_delta) = self.sort_wolves(&population, &fitness);
            let new_alpha_score = self.objective_fn.evaluate(&new_alpha);

            // Update best solution
            if new_alpha_score < alpha_score {
                alpha = new_alpha;
                beta = new_beta;
                delta = new_delta;
                alpha_score = new_alpha_score;

                if new_alpha_score < best_fitness {
                    best_solution = alpha.clone();
                    best_fitness = new_alpha_score;
                }
            }

            // Check termination condition
            if let Some(target) = self.params.opt_params.target_value {
                if best_fitness <= target {
                    final_iter = t + 1;
                    break;
                }
            }
            final_iter = t + 1;
        }

        OptimizationResult {
            best_solution,
            best_fitness,
            iterations: final_iter,
            evaluations: final_iter * pop_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_function::{Sphere, Beale, Rosenbrock, TestFunction};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_acgwo_sphere() {
        let sphere = Sphere::new(2);
        let params = ACGWOParams {
            opt_params: OptimizationParams {
                population_size: 50,
                max_iterations: 300,
                target_value: Some(1e-4),
            },
            lambda: 0.7,
            a_decay: 0.005,
        };

        let optimizer = ACGWO::new(&sphere, params);
        let result = optimizer.optimize();

        // Test convergence
        assert!(result.best_fitness < 1e-4);
        
        // Test solution is close to global minimum
        let global_min = sphere.global_minimum_position();
        for (x, x_min) in result.best_solution.iter().zip(global_min.iter()) {
            assert_abs_diff_eq!(*x, *x_min, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_acgwo_beale() {
        let beale = Beale::new();
        let params = ACGWOParams {
            opt_params: OptimizationParams {
                population_size: 50,
                max_iterations: 200,
                target_value: Some(1e-4),
            },
            lambda: 0.5,
            a_decay: 0.01,
        };

        let optimizer = ACGWO::new(&beale, params);
        let result = optimizer.optimize();

        // Test reasonable convergence
        assert!(result.best_fitness < 1.0);
        
        // Test solution is in reasonable range of global minimum
        let global_min = beale.global_minimum_position();
        for (x, x_min) in result.best_solution.iter().zip(global_min.iter()) {
            assert_abs_diff_eq!(*x, *x_min, epsilon = 0.5);
        }
    }

    #[test]
    fn test_acgwo_rosenbrock() {
        let rosenbrock = Rosenbrock::new(2);
        let params = ACGWOParams {
            opt_params: OptimizationParams {
                population_size: 50,
                max_iterations: 200,
                target_value: Some(1e-2),
            },
            lambda: 0.5,
            a_decay: 0.01,
        };

        let optimizer = ACGWO::new(&rosenbrock, params);
        let result = optimizer.optimize();

        // Test reasonable convergence for this difficult function
        assert!(result.best_fitness < 10.0);
    }

    #[test]
    fn test_acgwo_parameter_sensitivity() {
        let sphere = Sphere::new(2);
        
        // Test different population sizes
        let params1 = ACGWOParams {
            opt_params: OptimizationParams {
                population_size: 10,
                max_iterations: 100,
                target_value: None,
            },
            lambda: 0.5,
            a_decay: 0.01,
        };
        
        let params2 = ACGWOParams {
            opt_params: OptimizationParams {
                population_size: 100,
                max_iterations: 100,
                target_value: None,
            },
            lambda: 0.5,
            a_decay: 0.01,
        };

        let optimizer1 = ACGWO::new(&sphere, params1);
        let optimizer2 = ACGWO::new(&sphere, params2);
        
        let result1 = optimizer1.optimize();
        let result2 = optimizer2.optimize();

        // Larger population should generally give better results
        assert!(result2.best_fitness <= result1.best_fitness);
    }

    #[test]
    fn test_acgwo_chaotic_map() {
        let sphere = Sphere::new(2);
        let optimizer = ACGWO::new(
            &sphere,
            ACGWOParams::default()
        );

        // Test chaotic map properties
        let x = 0.3;
        let y = optimizer.chaotic_map(x);
        assert!(y >= 0.0 && y <= 1.0);

        let x = 0.7;
        let y = optimizer.chaotic_map(x);
        assert!(y >= 0.0 && y <= 1.0);
    }

    #[test]
    fn test_acgwo_early_stopping() {
        let sphere = Sphere::new(2);
        let params = ACGWOParams {
            opt_params: OptimizationParams {
                population_size: 50,
                max_iterations: 1000,
                target_value: Some(1e-6),
            },
            lambda: 0.7,
            a_decay: 0.005,
        };

        let optimizer = ACGWO::new(&sphere, params);
        let result = optimizer.optimize();

        // Should stop before max iterations if target value is reached
        assert!(result.iterations < 1000);
        assert!(result.best_fitness <= 1e-6);
    }
}
