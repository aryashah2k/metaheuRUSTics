use ndarray::{s, Array1};
use std::f64::consts::PI;
use super::TestFunction;
use crate::algorithm::ObjectiveFunction;

/// Griewank Function
/// f(x) = 1 + ∑(x²/4000) - ∏cos(x/√i)
#[derive(Debug)]
pub struct Griewank {
    dims: usize,
}

impl Griewank {
    pub fn new(dimensions: usize) -> Self {
        Self { dims: dimensions }
    }
}

impl ObjectiveFunction for Griewank {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        let sum = x.iter().map(|&xi| xi * xi / 4000.0).sum::<f64>();
        let prod = x.iter().enumerate()
            .map(|(i, &xi)| (xi / ((i + 1) as f64).sqrt()).cos())
            .product::<f64>();
        
        1.0 + sum - prod
    }
    
    fn dimensions(&self) -> usize {
        self.dims
    }
    
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![-600.0; self.dims],
            vec![600.0; self.dims]
        )
    }
}

impl TestFunction for Griewank {
    fn global_minimum(&self) -> f64 {
        0.0
    }
    
    fn global_minimum_position(&self) -> Array1<f64> {
        Array1::zeros(self.dims)
    }
    
    fn name(&self) -> &'static str {
        "Griewank Function"
    }
}

/// Schwefel Function
/// f(x) = 418.9829n - ∑(x_i * sin(√|x_i|))
#[derive(Debug)]
pub struct Schwefel {
    dims: usize,
}

impl Schwefel {
    pub fn new(dimensions: usize) -> Self {
        Self { dims: dimensions }
    }
}

impl ObjectiveFunction for Schwefel {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        418.9829 * self.dims as f64
            - x.iter()
                .map(|&xi| xi * (xi.abs().sqrt()).sin())
                .sum::<f64>()
    }
    
    fn dimensions(&self) -> usize {
        self.dims
    }
    
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![-500.0; self.dims],
            vec![500.0; self.dims]
        )
    }
}

impl TestFunction for Schwefel {
    fn global_minimum(&self) -> f64 {
        0.0
    }
    
    fn global_minimum_position(&self) -> Array1<f64> {
        Array1::from_elem(self.dims, 420.9687)
    }
    
    fn name(&self) -> &'static str {
        "Schwefel Function"
    }
}

/// Levy Function
/// f(x) = sin²(πw₁) + ∑(w_i-1)²[1+10sin²(πw_i+1)] + (w_n-1)²[1+sin²(2πw_n)]
/// where w_i = 1 + (x_i-1)/4
#[derive(Debug)]
pub struct Levy {
    dims: usize,
}

impl Levy {
    pub fn new(dimensions: usize) -> Self {
        Self { dims: dimensions }
    }
}

impl ObjectiveFunction for Levy {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        let w: Array1<f64> = x.mapv(|xi| 1.0 + (xi - 1.0) / 4.0);
        
        let term1 = (PI * w[0]).sin().powi(2);
        
        let term2 = w.slice(s![..-1])
            .iter()
            .map(|&wi| {
                (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + 1.0).sin().powi(2))
            })
            .sum::<f64>();
        
        let n = w.len() - 1;
        let term3 = (w[n] - 1.0).powi(2) * (1.0 + (2.0 * PI * w[n]).sin().powi(2));
        
        term1 + term2 + term3
    }
    
    fn dimensions(&self) -> usize {
        self.dims
    }
    
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![-10.0; self.dims],
            vec![10.0; self.dims]
        )
    }
}

impl TestFunction for Levy {
    fn global_minimum(&self) -> f64 {
        0.0
    }
    
    fn global_minimum_position(&self) -> Array1<f64> {
        Array1::ones(self.dims)
    }
    
    fn name(&self) -> &'static str {
        "Levy Function"
    }
}

/// Michalewicz Function
/// f(x) = -∑sin(x_i)[sin(ix_i²/π)]^(2m)
#[derive(Debug)]
pub struct Michalewicz {
    dims: usize,
    m: f64,
}

impl Michalewicz {
    pub fn new(dimensions: usize, m: f64) -> Self {
        Self { dims: dimensions, m }
    }
}

impl ObjectiveFunction for Michalewicz {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        -x.iter().enumerate()
            .map(|(i, &xi)| {
                xi.sin() * ((i + 1) as f64 * xi * xi / PI).sin().powf(2.0 * self.m)
            })
            .sum::<f64>()
    }
    
    fn dimensions(&self) -> usize {
        self.dims
    }
    
    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.0; self.dims],
            vec![PI; self.dims]
        )
    }
}

impl TestFunction for Michalewicz {
    fn global_minimum(&self) -> f64 {
        match self.dims {
            2 => -1.8013,
            5 => -4.687658,
            10 => -9.66015,
            _ => f64::NEG_INFINITY, // Exact value depends on dimensions
        }
    }
    
    fn global_minimum_position(&self) -> Array1<f64> {
        // Position depends on dimensions and m parameter
        // Here we return a reasonable approximation
        Array1::from_elem(self.dims, 2.20)
    }
    
    fn name(&self) -> &'static str {
        "Michalewicz Function"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_griewank_minimum() {
        let griewank = Griewank::new(2);
        let x = Array1::zeros(2);
        assert_relative_eq!(griewank.evaluate(&x), 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_schwefel_bounds() {
        let schwefel = Schwefel::new(2);
        let (lower, upper) = schwefel.bounds();
        assert_eq!(lower[0], -500.0);
        assert_eq!(upper[0], 500.0);
    }
    
    #[test]
    fn test_levy_minimum() {
        let levy = Levy::new(2);
        let x = Array1::ones(2);
        assert_relative_eq!(levy.evaluate(&x), 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_michalewicz_bounds() {
        let michalewicz = Michalewicz::new(2, 10.0);
        let (lower, upper) = michalewicz.bounds();
        assert_eq!(lower[0], 0.0);
        assert_relative_eq!(upper[0], PI, epsilon = 1e-10);
    }
}
