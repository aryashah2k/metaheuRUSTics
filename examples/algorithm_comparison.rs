use metaheurustics::prelude::*;
use std::{time::Instant, fs};

fn run_optimization<T: TestFunction + ObjectiveFunction>(
    problem: &T,
    opt_params: &OptimizationParams,
    plot_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nProblem: {}", problem.name());
    println!("Known global minimum: {}", problem.global_minimum());
    println!("Dimensions: {}", problem.dimensions());
    println!("\nResults:");
    
    // PSO
    let pso_params = PSOParams {
        opt_params: opt_params.clone(),
        w: 0.9,
        c1: 2.0,
        c2: 2.0,
        w_decay: 0.01,
    };
    let start = Instant::now();
    let pso_result = PSO::new(problem, pso_params).optimize()?;
    let pso_time = start.elapsed();
    
    // DE
    let de_params = DEParams {
        opt_params: opt_params.clone(),
        mutation_factor: 0.8,
        crossover_rate: 0.9,
        strategy: DEStrategy::RandOneBin,
    };
    let start = Instant::now();
    let de_result = DE::new(problem, de_params).optimize()?;
    let de_time = start.elapsed();
    
    // ABCO
    let abco_params = ABCOParams {
        opt_params: opt_params.clone(),
        limit: 100,
    };
    let start = Instant::now();
    let abco_result = ABCO::new(problem, abco_params).optimize()?;
    let abco_time = start.elapsed();
    
    // GWO
    let gwo_params = GWOParams {
        opt_params: opt_params.clone(),
        a_decay: 2.0,
    };
    let start = Instant::now();
    let gwo_result = GWO::new(problem, gwo_params).optimize()?;
    let gwo_time = start.elapsed();
    
    // Print results
    println!("\nOptimization Results:");
    println!("PSO:");
    println!("  Best fitness: {}", pso_result.best_fitness);
    println!("  Best solution: {:?}", pso_result.best_solution);
    println!("  Time: {:?}", pso_time);
    
    println!("\nDE:");
    println!("  Best fitness: {}", de_result.best_fitness);
    println!("  Best solution: {:?}", de_result.best_solution);
    println!("  Time: {:?}", de_time);
    
    println!("\nABCO:");
    println!("  Best fitness: {}", abco_result.best_fitness);
    println!("  Best solution: {:?}", abco_result.best_solution);
    println!("  Time: {:?}", abco_time);
    
    println!("\nGWO:");
    println!("  Best fitness: {}", gwo_result.best_fitness);
    println!("  Best solution: {:?}", gwo_result.best_solution);
    println!("  Time: {:?}", gwo_time);
    
    // Plot results if directory is provided
    if let Some(plot_dir) = plot_dir {
        // Create plot directory if it doesn't exist
        fs::create_dir_all(plot_dir)?;
        
        // Plot PSO result
        let filename = format!("{}/{}_pso.png", plot_dir, problem.name().to_lowercase().replace(" ", "_"));
        plot_2d_landscape(problem, Some(&pso_result.best_solution), &filename)?;
        
        // Plot DE result
        let filename = format!("{}/{}_de.png", plot_dir, problem.name().to_lowercase().replace(" ", "_"));
        plot_2d_landscape(problem, Some(&de_result.best_solution), &filename)?;
        
        // Plot ABCO result
        let filename = format!("{}/{}_abco.png", plot_dir, problem.name().to_lowercase().replace(" ", "_"));
        plot_2d_landscape(problem, Some(&abco_result.best_solution), &filename)?;
        
        // Plot GWO result
        let filename = format!("{}/{}_gwo.png", plot_dir, problem.name().to_lowercase().replace(" ", "_"));
        plot_2d_landscape(problem, Some(&gwo_result.best_solution), &filename)?;
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create optimization problems
    let sphere = Sphere::new(2);
    let ackley = Ackley::new(2);
    let rosenbrock = Rosenbrock::new(2);
    let rastrigin = Rastrigin::new(2);
    
    // Common optimization parameters
    let opt_params = OptimizationParams {
        population_size: 30,
        max_iterations: 100,
        target_value: Some(1e-4),
    };
    
    // Get plot directory if set
    let plot_dir = std::env::var_os("PLOT_DIR").map(|p| p.to_string_lossy().into_owned()).unwrap_or("plots".to_string());
    
    // Run optimizations for each problem
    run_optimization(&sphere, &opt_params, Some(&plot_dir))?;
    run_optimization(&ackley, &opt_params, Some(&plot_dir))?;
    run_optimization(&rosenbrock, &opt_params, Some(&plot_dir))?;
    run_optimization(&rastrigin, &opt_params, Some(&plot_dir))?;
    
    println!("\nPlots have been saved to the '{}' directory", plot_dir);
    
    Ok(())
}
