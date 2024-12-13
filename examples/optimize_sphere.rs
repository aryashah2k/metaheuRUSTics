use metaheurustics::prelude::*;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2D sphere optimization problem
    let sphere = Sphere::new(2);
    
    // Configure PSO parameters
    let params = PSOParams {
        opt_params: OptimizationParams {
            population_size: 30,
            max_iterations: 100,
            target_value: Some(1e-4),
        },
        w: 0.9,
        c1: 2.0,
        c2: 2.0,
        w_decay: 0.01,
    };
    
    // Print problem info
    println!("Problem: {}", sphere.name());
    println!("Known global minimum: {}", sphere.global_minimum());
    
    // Create and run optimizer
    let optimizer = PSO::new(&sphere, params);
    let result = optimizer.optimize()?;
    
    println!("\nOptimization Results:");
    println!("Best fitness: {}", result.best_fitness);
    println!("Best solution: {:?}", result.best_solution);
    println!("Number of iterations: {}", result.iterations);
    println!("Number of function evaluations: {}", result.evaluations);
    
    // Create plots directory
    let plot_dir = "plots";
    fs::create_dir_all(plot_dir)?;
    
    // Plot the optimization landscape with the final solution
    // 2D contour plot
    let filename = format!("{}/sphere_pso_contour.png", plot_dir);
    plot_2d_landscape(&sphere, Some(&result.best_solution), &filename)?;
    
    // 3D surface plot
    let filename = format!("{}/sphere_pso_surface.png", plot_dir);
    plot_3d_landscape(&sphere, Some(&result.best_solution), &filename)?;
    
    println!("\nPlots have been saved to the '{}' directory:", plot_dir);
    println!("- 2D contour plot: sphere_pso_contour.png");
    println!("- 3D surface plot: sphere_pso_surface.png");
    
    Ok(())
}
