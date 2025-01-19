use ndarray::Array1;
use rand::Rng;
use std::f64;

pub struct FireflyAlgorithm {
    pub population_size: usize,
    pub max_iterations: usize,
    pub alpha: f64,  // light absorption coefficient
    pub beta0: f64,  // attractiveness at distance 0
    pub gamma: f64,  // light absorption coefficient
}

impl Default for FireflyAlgorithm {
    fn default() -> Self {
        FireflyAlgorithm {
            population_size: 30,
            max_iterations: 1000,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 1.0,
        }
    }
}

impl FireflyAlgorithm {
    pub fn new(population_size: usize, max_iterations: usize) -> Self {
        FireflyAlgorithm {
            population_size,
            max_iterations,
            ..Default::default()
        }
    }

    pub fn optimize<F>(&self, objective_fn: F, bounds: &[(f64, f64)]) -> Array1<f64>
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let dim = bounds.len();
        let mut rng = rand::thread_rng();
        
        // Initialize population
        let mut population: Vec<(Array1<f64>, f64)> = (0..self.population_size)
            .map(|_| {
                let position = Array1::from_vec(
                    bounds.iter()
                        .map(|(min, max)| rng.gen_range(*min..*max))
                        .collect()
                );
                let fitness = objective_fn(&position);
                (position, fitness)
            })
            .collect();

        let mut best_solution = population
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .clone();

        // Main iteration loop
        for _ in 0..self.max_iterations {
            // For each firefly
            for i in 0..self.population_size {
                // Compare with all other fireflies
                for j in 0..self.population_size {
                    if population[j].1 < population[i].1 {
                        // Calculate distance
                        let r = (&population[i].0 - &population[j].0)
                            .mapv(|x| x * x)
                            .sum()
                            .sqrt();

                        // Calculate attractiveness
                        let beta = self.beta0 * (-self.gamma * r * r).exp();

                        // Update position
                        let mut new_position = Array1::zeros(dim);
                        for d in 0..dim {
                            let rand_step = rng.gen_range(-0.5..0.5);
                            new_position[d] = population[i].0[d] +
                                beta * (population[j].0[d] - population[i].0[d]) +
                                self.alpha * rand_step;
                            
                            // Bound checking
                            if new_position[d] < bounds[d].0 {
                                new_position[d] = bounds[d].0;
                            }
                            if new_position[d] > bounds[d].1 {
                                new_position[d] = bounds[d].1;
                            }
                        }

                        let new_fitness = objective_fn(&new_position);
                        if new_fitness < population[i].1 {
                            population[i] = (new_position, new_fitness);
                        }
                    }
                }
            }

            // Update best solution
            if let Some(current_best) = population
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                if current_best.1 < best_solution.1 {
                    best_solution = current_best.clone();
                }
            }
        }

        best_solution.0
    }
}
