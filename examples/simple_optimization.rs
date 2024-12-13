use metaheurustics::{
    algorithm::{
        ga::{GA, GAParams},
        sa::{SA, SAParams},
        de::{DE, DEParams, DEStrategy},
        pso::{PSO, PSOParams},
        OptimizationParams,
    },
    test_function::Rosenbrock,
    utils::{plot_2d_landscape, plot_3d_landscape},
};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create test function
    let rosenbrock = Rosenbrock::new(2);
    
    // Set common optimization parameters
    let opt_params = OptimizationParams {
        population_size: 50,
        max_iterations: 1000,
        target_value: Some(1e-6),
    };

    // Create plots directory
    let plot_dir = "plots";
    fs::create_dir_all(plot_dir)?;

    // Run GA
    println!("\nRunning Genetic Algorithm...");
    let ga_params = GAParams {
        opt_params: opt_params.clone(),
        crossover_rate: 0.8,
        mutation_rate: 0.1,
        tournament_size: 3,
    };
    let ga = GA::new(&rosenbrock, ga_params);
    match ga.optimize() {
        Ok(result) => {
            println!("GA Result: {:?}", result);
            let filename = format!("{}/rosenbrock_ga_contour.png", plot_dir);
            plot_2d_landscape(&rosenbrock, Some(&result.best_solution), &filename)?;
            let filename = format!("{}/rosenbrock_ga_surface.png", plot_dir);
            plot_3d_landscape(&rosenbrock, Some(&result.best_solution), &filename)?;
        },
        Err(e) => println!("GA Error: {:?}", e),
    }

    // Run PSO
    println!("\nRunning Particle Swarm Optimization...");
    let pso_params = PSOParams {
        opt_params: opt_params.clone(),
        w: 0.7,
        c1: 2.0,
        c2: 2.0,
        w_decay: 0.99,
    };
    let pso = PSO::new(&rosenbrock, pso_params);
    match pso.optimize() {
        Ok(result) => {
            println!("PSO Result: {:?}", result);
            let filename = format!("{}/rosenbrock_pso_contour.png", plot_dir);
            plot_2d_landscape(&rosenbrock, Some(&result.best_solution), &filename)?;
            let filename = format!("{}/rosenbrock_pso_surface.png", plot_dir);
            plot_3d_landscape(&rosenbrock, Some(&result.best_solution), &filename)?;
        },
        Err(e) => println!("PSO Error: {:?}", e),
    }

    // Run DE
    println!("\nRunning Differential Evolution...");
    let de_params = DEParams {
        opt_params: opt_params.clone(),
        mutation_factor: 0.8,
        crossover_rate: 0.9,
        strategy: DEStrategy::RandOneBin,
    };
    let de = DE::new(&rosenbrock, de_params);
    match de.optimize() {
        Ok(result) => {
            println!("DE Result: {:?}", result);
            let filename = format!("{}/rosenbrock_de_contour.png", plot_dir);
            plot_2d_landscape(&rosenbrock, Some(&result.best_solution), &filename)?;
            let filename = format!("{}/rosenbrock_de_surface.png", plot_dir);
            plot_3d_landscape(&rosenbrock, Some(&result.best_solution), &filename)?;
        },
        Err(e) => println!("DE Error: {:?}", e),
    }

    // Run SA
    println!("\nRunning Simulated Annealing...");
    let sa_params = SAParams {
        opt_params: opt_params.clone(),
        initial_temp: 100.0,
        cooling_rate: 0.95,
        iterations_per_temp: 50,
        min_temp: 1e-10,
    };
    let sa = SA::new(&rosenbrock, sa_params);
    match sa.optimize() {
        Ok(result) => {
            println!("SA Result: {:?}", result);
            let filename = format!("{}/rosenbrock_sa_contour.png", plot_dir);
            plot_2d_landscape(&rosenbrock, Some(&result.best_solution), &filename)?;
            let filename = format!("{}/rosenbrock_sa_surface.png", plot_dir);
            plot_3d_landscape(&rosenbrock, Some(&result.best_solution), &filename)?;
        },
        Err(e) => println!("SA Error: {:?}", e),
    }

    println!("\nPlots have been saved to the '{}' directory", plot_dir);
    println!("Each algorithm has both contour and surface plots.");
    
    Ok(())
}
