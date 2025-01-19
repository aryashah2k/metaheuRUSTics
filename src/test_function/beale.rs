use ndarray::Array1;
use crate::algorithm::ObjectiveFunction;
use crate::test_function::TestFunction;

/// Beale Function
/// f(x₁,x₂) = (1.5 - x₁ + x₁x₂)² + (2.25 - x₁ + x₁x₂²)² + (2.625 - x₁ + x₁x₂³)²
/// Global minimum: f(3, 0.5) = 0
#[derive(Debug)]
pub struct Beale;

impl Beale {
    /// Create a new instance of the Beale function
    pub fn new() -> Self {
        Beale
    }
}

impl ObjectiveFunction for Beale {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        if x.len() != 2 {
            panic!("Beale function is only defined for 2 dimensions");
        }

        let x1 = x[0];
        let x2 = x[1];
        
        let term1 = (1.5 - x1 + x1 * x2).powi(2);
        let term2 = (2.25 - x1 + x1 * x2.powi(2)).powi(2);
        let term3 = (2.625 - x1 + x1 * x2.powi(3)).powi(2);
        
        term1 + term2 + term3
    }

    fn dimensions(&self) -> usize {
        2
    }

    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![-4.5; 2], vec![4.5; 2])
    }
}

impl TestFunction for Beale {
    fn global_minimum(&self) -> f64 {
        0.0
    }

    fn global_minimum_position(&self) -> Array1<f64> {
        Array1::from_vec(vec![3.0, 0.5])
    }

    fn name(&self) -> &'static str {
        "Beale"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_beale_global_minimum() {
        let beale = Beale::new();
        let global_min_pos = beale.global_minimum_position();
        let global_min = beale.evaluate(&global_min_pos);
        assert_abs_diff_eq!(global_min, beale.global_minimum(), epsilon = 1e-6);
    }

    #[test]
    fn test_beale_known_points() {
        let beale = Beale::new();
        
        // Test origin
        let origin = Array1::zeros(2);
        assert_abs_diff_eq!(beale.evaluate(&origin), 14.203125, epsilon = 1e-6);
        
        // Test point (1, 1)
        let point1 = Array1::from_vec(vec![1.0, 1.0]);
        assert_abs_diff_eq!(beale.evaluate(&point1), 14.203125, epsilon = 1e-6);
        
        // Test point (2, 2)
        let point2 = Array1::from_vec(vec![2.0, 2.0]);
        assert_abs_diff_eq!(beale.evaluate(&point2), 356.703125, epsilon = 1e-6);
    }

    #[test]
    fn test_beale_bounds() {
        let beale = Beale::new();
        let (lower, upper) = beale.bounds();
        
        // Test bounds are correct
        assert_eq!(lower, vec![-4.5; 2]);
        assert_eq!(upper, vec![4.5; 2]);
        
        // Test points at bounds
        let lower_bound = Array1::from_vec(lower.clone());
        let upper_bound = Array1::from_vec(upper.clone());
        
        // Ensure function can be evaluated at bounds
        let _ = beale.evaluate(&lower_bound);
        let _ = beale.evaluate(&upper_bound);
    }

    #[test]
    fn test_beale_dimensions() {
        let beale = Beale::new();
        assert_eq!(beale.dimensions(), 2);
    }

    #[test]
    #[should_panic(expected = "Beale function is only defined for 2 dimensions")]
    fn test_beale_wrong_dimensions() {
        let beale = Beale::new();
        let wrong_dim = Array1::zeros(3);
        beale.evaluate(&wrong_dim);
    }

    #[test]
    fn test_beale_symmetry() {
        let beale = Beale::new();
        let point1 = Array1::from_vec(vec![1.0, 2.0]);
        let point2 = Array1::from_vec(vec![1.0, -2.0]);
        
        // Beale function is not symmetric
        assert_ne!(beale.evaluate(&point1), beale.evaluate(&point2));
    }
}
