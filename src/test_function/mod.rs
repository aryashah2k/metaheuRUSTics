use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use crate::algorithm::ObjectiveFunction;

mod additional;
pub use additional::*;

mod beale;
pub use beale::*;

/// Common trait for test functions
pub trait TestFunction: ObjectiveFunction {
    /// Get the known global minimum value
    fn global_minimum(&self) -> f64;
    
    /// Get the known global minimum position
    fn global_minimum_position(&self) -> Array1<f64>;
    
    /// Get the name of the test function
    fn name(&self) -> &'static str;

    fn test_3d_surface(&self, n_points: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let (min_bounds, max_bounds) = self.bounds();
        let x_min = min_bounds[0];
        let x_max = max_bounds[0];
        let y_min = min_bounds[1];
        let y_max = max_bounds[1];

        let x_step = (x_max - x_min) / (n_points - 1) as f64;
        let y_step = (y_max - y_min) / (n_points - 1) as f64;

        let mut x = Array2::zeros((n_points, n_points));
        let mut y = Array2::zeros((n_points, n_points));
        let mut z = Array2::zeros((n_points, n_points));

        for i in 0..n_points {
            for j in 0..n_points {
                let x_val = x_min + i as f64 * x_step;
                let y_val = y_min + j as f64 * y_step;
                x[[i, j]] = x_val;
                y[[i, j]] = y_val;
                z[[i, j]] = self.evaluate(&Array1::from_vec(vec![x_val, y_val]));
            }
        }

        (x, y, z)
    }
}

/// Ackley Function
/// f(x) = -20exp(-0.2√(1/n∑x²)) - exp(1/n∑cos(2πx)) + 20 + e
#[derive(Debug)]
pub struct Ackley {
    dims: usize,
}

impl Ackley {
    pub fn new(dimensions: usize) -> Self {
        Self { dims: dimensions }
    }
}

impl ObjectiveFunction for Ackley {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        let a = 20.0;
        let b = 0.2;
        let c = 2.0 * PI;
        
        let n = x.len() as f64;
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let sum_cos = x.iter().map(|xi| (c * xi).cos()).sum::<f64>();
        
        -a * (-b * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + a + std::f64::consts::E
    }
    
    fn dimensions(&self) -> usize {
        self.dims
    }
    
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![-32.768; self.dims],
            vec![32.768; self.dims]
        )
    }
}

impl TestFunction for Ackley {
    fn global_minimum(&self) -> f64 {
        0.0
    }
    
    fn global_minimum_position(&self) -> Array1<f64> {
        Array1::zeros(self.dims)
    }
    
    fn name(&self) -> &'static str {
        "Ackley Function"
    }
}

/// Sphere Function (De Jong F1)
/// f(x) = ∑x²
#[derive(Debug)]
pub struct Sphere {
    dims: usize,
}

impl Sphere {
    pub fn new(dimensions: usize) -> Self {
        Self { dims: dimensions }
    }
}

impl ObjectiveFunction for Sphere {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
    
    fn dimensions(&self) -> usize {
        self.dims
    }
    
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![-5.12; self.dims],
            vec![5.12; self.dims]
        )
    }
}

impl TestFunction for Sphere {
    fn global_minimum(&self) -> f64 {
        0.0
    }
    
    fn global_minimum_position(&self) -> Array1<f64> {
        Array1::zeros(self.dims)
    }
    
    fn name(&self) -> &'static str {
        "Sphere Function"
    }
}

/// Rosenbrock Function (De Jong F2)
/// f(x) = ∑[100(x_{i+1} - x_i²)² + (1 - x_i)²]
#[derive(Debug)]
pub struct Rosenbrock {
    dims: usize,
}

impl Rosenbrock {
    pub fn new(dimensions: usize) -> Self {
        if dimensions < 2 {
            panic!("Rosenbrock function requires at least 2 dimensions");
        }
        Self { dims: dimensions }
    }
}

impl ObjectiveFunction for Rosenbrock {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.dims - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (x[i] - 1.0).powi(2);
        }
        sum
    }
    
    fn dimensions(&self) -> usize {
        self.dims
    }
    
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![-2.048; self.dims],
            vec![2.048; self.dims]
        )
    }
}

impl TestFunction for Rosenbrock {
    fn global_minimum(&self) -> f64 {
        0.0
    }
    
    fn global_minimum_position(&self) -> Array1<f64> {
        Array1::ones(self.dims)
    }
    
    fn name(&self) -> &'static str {
        "Rosenbrock Function"
    }
}

/// Rastrigin Function
/// f(x) = 10n + ∑[x² - 10cos(2πx)]
#[derive(Debug)]
pub struct Rastrigin {
    dims: usize,
}

impl Rastrigin {
    pub fn new(dimensions: usize) -> Self {
        Self { dims: dimensions }
    }
}

impl ObjectiveFunction for Rastrigin {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        let sum = x.iter()
            .map(|xi| xi * xi - a * (2.0 * PI * xi).cos())
            .sum::<f64>();
        a * n + sum
    }
    
    fn dimensions(&self) -> usize {
        self.dims
    }
    
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![-5.12; self.dims],
            vec![5.12; self.dims]
        )
    }
}

impl TestFunction for Rastrigin {
    fn global_minimum(&self) -> f64 {
        0.0
    }
    
    fn global_minimum_position(&self) -> Array1<f64> {
        Array1::zeros(self.dims)
    }
    
    fn name(&self) -> &'static str {
        "Rastrigin Function"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_sphere_function() {
        let sphere = Sphere::new(2);
        let x = Array1::zeros(2);
        assert_eq!(sphere.evaluate(&x), 0.0);
        
        let x = Array1::from_vec(vec![1.0, 1.0]);
        assert_eq!(sphere.evaluate(&x), 2.0);
    }
    
    #[test]
    fn test_ackley_function() {
        let ackley = Ackley::new(2);
        let x = Array1::zeros(2);
        assert_relative_eq!(ackley.evaluate(&x), 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_rosenbrock_function() {
        let rosenbrock = Rosenbrock::new(2);
        let x = Array1::ones(2);
        assert_eq!(rosenbrock.evaluate(&x), 0.0);
    }
    
    #[test]
    fn test_rastrigin_function() {
        let rastrigin = Rastrigin::new(2);
        let x = Array1::zeros(2);
        assert_eq!(rastrigin.evaluate(&x), 0.0);
    }
}
