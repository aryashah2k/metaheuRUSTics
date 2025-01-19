use ndarray::{arr1, Array1, Array2};
use plotters::{
    prelude::*,
    style::{Color, RGBColor, HSLColor},
};
use metaheurustics::{
    algorithm::{
        pso::{PSO, PSOParams},
        de::{DE, DEParams, DEStrategy},
        ga::{GA, GAParams},
        sa::{SA, SAParams},
        acgwo::{ACGWO, ACGWOParams},
        abco::{ABCO, ABCOParams},
        gwo::{GWO, GWOParams},
        fa::FireflyAlgorithm,
        OptimizationParams,
    },
    prelude::*,
    MetaheuristicError,
};
use metaheurustics::test_function::{
    Sphere,
    Ackley,
    Rosenbrock,
    Rastrigin,
    Beale,
    Griewank,
    TestFunction,
};

fn create_3d_surface_plot<F: TestFunction>(
    test_function: &F,
    result: &OptimizationResult,
    algorithm_name: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create directory for plots if it doesn't exist
    std::fs::create_dir_all("plots")?;

    let n = 50;
    let x = Array1::linspace(-10.0, 10.0, n);
    let y = Array1::linspace(-10.0, 10.0, n);
    let mut z = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let point = arr1(&[x[i], y[j]]);
            z[[i, j]] = test_function.evaluate(&point);
        }
    }

    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - {} 3D Surface Plot", test_function.name(), algorithm_name),
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-10f64..10f64, -10f64..10f64)?;

    chart.configure_mesh().draw()?;

    // Plot surface as a heatmap
    let min_z = z.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_z = z.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    for i in 0..n-1 {
        for j in 0..n-1 {
            let z_val = z[[i, j]];
            let color_scale = (z_val - min_z) / (max_z - min_z);
            let color = HSLColor(
                240.0 / 360.0 - 240.0 / 360.0 * color_scale,
                0.8,
                0.3 + 0.4 * color_scale,
            );

            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (x[i], y[j]),
                    (x[i + 1], y[j + 1]),
                ],
                color.filled(),
            )))?;
        }
    }

    chart.draw_series(PointSeries::of_element(
        vec![(result.best_solution[0], result.best_solution[1])],
        5,
        &RED,
        &|c, s, st| {
            EmptyElement::at(c)
                + Circle::new((0, 0), s, st.filled())
                + Text::new(
                    format!("Best Solution"),
                    (10, 0),
                    ("sans-serif", 15).into_font(),
                )
        },
    ))?;

    root.present()?;
    println!("3D surface plot saved as '{}'", filename);
    Ok(())
}

fn create_contour_plot<F: TestFunction>(
    test_function: &F,
    result: &OptimizationResult,
    algorithm_name: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = 50;
    let x = Array1::linspace(-10.0, 10.0, n);
    let y = Array1::linspace(-10.0, 10.0, n);
    let mut z = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let point = arr1(&[x[i], y[j]]);
            z[[i, j]] = test_function.evaluate(&point);
        }
    }

    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - {} Contour Plot", test_function.name(), algorithm_name),
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-10f64..10f64, -10f64..10f64)?;

    chart.configure_mesh().draw()?;

    let min_z = z.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_z = z.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let levels = (0..10).map(|i| min_z + (max_z - min_z) * i as f64 / 9.0);

    for level in levels {
        let mut points = Vec::new();
        for i in 0..n-1 {
            for j in 0..n-1 {
                let z00 = z[[i, j]];
                let z01 = z[[i, j+1]];
                let z10 = z[[i+1, j]];
                let z11 = z[[i+1, j+1]];
                
                if (z00 <= level && level < z01) || (z01 <= level && level < z00) ||
                   (z10 <= level && level < z11) || (z11 <= level && level < z10) ||
                   (z00 <= level && level < z10) || (z10 <= level && level < z00) ||
                   (z01 <= level && level < z11) || (z11 <= level && level < z01) {
                    points.push((x[i], y[j]));
                }
            }
        }

        chart.draw_series(LineSeries::new(
            points,
            &RGBColor(0, 0, 255).mix(0.3),
        ))?;
    }

    chart.draw_series(PointSeries::of_element(
        vec![(result.best_solution[0], result.best_solution[1])],
        5,
        &RED,
        &|c, s, st| {
            EmptyElement::at(c)
                + Circle::new((0, 0), s, st.filled())
                + Text::new(
                    format!("Best Solution"),
                    (10, 0),
                    ("sans-serif", 15).into_font(),
                )
        },
    ))?;

    root.present()?;
    println!("Contour plot saved as '{}'", output_path);
    Ok(())
}

fn optimize_and_plot<F: TestFunction>(
    test_function: &F,
    algorithm_name: &str,
    run_optimization: impl Fn(&F) -> Result<OptimizationResult, MetaheuristicError>,
) -> Result<(), Box<dyn std::error::Error>> {
    let result = run_optimization(test_function)?;
    
    let base_name = format!(
        "plots/{}_{}_",
        test_function.name().to_lowercase(),
        algorithm_name.to_lowercase()
    );

    // Create 3D surface plot
    create_3d_surface_plot(
        test_function,
        &result,
        algorithm_name,
        &format!("{}surface.png", base_name),
    )?;

    // Create contour plot
    create_contour_plot(
        test_function,
        &result,
        algorithm_name,
        &format!("{}contour.png", base_name),
    )?;

    // Create convergence plot
    let convergence_plot_path = format!("{}convergence.png", base_name);
    let root = BitMapBackend::new(&convergence_plot_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "{} - {} Convergence Plot",
                test_function.name(),
                algorithm_name
            ),
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f64..result.iterations as f64,
            result.best_fitness..result.best_fitness + 1.0,
        )?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Best Fitness")
        .draw()?;

    chart.draw_series(LineSeries::new(
        vec![(0.0, result.best_fitness), (result.iterations as f64, result.best_fitness)],
        &RED,
    ))?;

    root.present()?;

    // Create solution space plot
    let solution_plot_path = format!("{}solution.png", base_name);
    let root = BitMapBackend::new(&solution_plot_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - {} Solution Space", test_function.name(), algorithm_name),
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(-10f64..10f64, -10f64..10f64)?;

    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()?;

    // Plot the best solution point
    chart.draw_series(PointSeries::of_element(
        vec![(result.best_solution[0], result.best_solution[1])],
        5,
        &RED,
        &|c, s, st| {
            return EmptyElement::at(c)
                + Circle::new((0, 0), s, st.filled())
                + Text::new(
                    format!("Best Solution"),
                    (10, 0),
                    ("sans-serif", 15).into_font(),
                );
        },
    ))?;

    root.present()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test functions
    let sphere = Sphere::new(2);
    let ackley = Ackley::new(2);
    let rosenbrock = Rosenbrock::new(2);
    let rastrigin = Rastrigin::new(2);
    let beale = Beale::new();
    let griewank = Griewank::new(2);

    // Common optimization parameters
    let opt_params = OptimizationParams {
        population_size: 50,
        max_iterations: 300,
        target_value: Some(1e-6),
    };

    // Algorithm configurations
    let pso_params = PSOParams {
        opt_params: opt_params.clone(),
        w: 0.7,
        c1: 2.0,
        c2: 2.0,
        w_decay: 0.99,
    };

    let de_params = DEParams {
        opt_params: opt_params.clone(),
        crossover_rate: 0.9,
        mutation_factor: 0.8,
        strategy: DEStrategy::RandOneBin,
    };

    let ga_params = GAParams {
        opt_params: opt_params.clone(),
        crossover_rate: 0.8,
        mutation_rate: 0.1,
        tournament_size: 3,
    };

    let sa_params = SAParams {
        opt_params: opt_params.clone(),
        initial_temp: 1.0,
        cooling_rate: 0.95,
        iterations_per_temp: 50,
        min_temp: 1e-10,
    };

    let acgwo_params = ACGWOParams {
        opt_params: opt_params.clone(),
        lambda: 0.7,
        a_decay: 0.005,
    };

    let abco_params = ABCOParams {
        opt_params: opt_params.clone(),
        limit: 20,
    };

    let gwo_params = GWOParams {
        opt_params: opt_params.clone(),
        a_decay: 0.01,
    };

    // Configure Firefly Algorithm
    let fa = FireflyAlgorithm::new(opt_params.population_size, opt_params.max_iterations);

    // Run optimization for each function and algorithm
    
    // Sphere
    if let Err(e) = optimize_and_plot(
        &sphere,
        "PSO",
        |f| PSO::new(f, pso_params.clone()).optimize(),
    ) {
        eprintln!("Error running PSO on Sphere: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &sphere,
        "DE",
        |f| DE::new(f, de_params.clone()).optimize(),
    ) {
        eprintln!("Error running DE on Sphere: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &sphere,
        "GA",
        |f| GA::new(f, ga_params.clone()).optimize(),
    ) {
        eprintln!("Error running GA on Sphere: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &sphere,
        "SA",
        |f| SA::new(f, sa_params.clone()).optimize(),
    ) {
        eprintln!("Error running SA on Sphere: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &sphere,
        "ACGWO",
        |f| Ok(ACGWO::new(f, acgwo_params.clone()).optimize()),
    ) {
        eprintln!("Error running ACGWO on Sphere: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &sphere,
        "ABCO",
        |f| ABCO::new(f, abco_params.clone()).optimize(),
    ) {
        eprintln!("Error running ABCO on Sphere: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &sphere,
        "GWO",
        |f| GWO::new(f, gwo_params.clone()).optimize(),
    ) {
        eprintln!("Error running GWO on Sphere: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &sphere,
        "FA",
        |f| {
            let (min_bounds, max_bounds) = f.bounds();
            let bounds: Vec<(f64, f64)> = min_bounds.into_iter()
                .zip(max_bounds.into_iter())
                .collect();
            let best_solution = fa.optimize(|x| f.evaluate(x), &bounds);
            let best_fitness = f.evaluate(&best_solution);
            Ok(OptimizationResult {
                best_solution,
                best_fitness,
                iterations: opt_params.max_iterations,
                evaluations: opt_params.max_iterations * opt_params.population_size,
            })
        },
    ) {
        eprintln!("Error running FA on Sphere: {}", e);
    }

    // Ackley
    if let Err(e) = optimize_and_plot(
        &ackley,
        "PSO",
        |f| PSO::new(f, pso_params.clone()).optimize(),
    ) {
        eprintln!("Error running PSO on Ackley: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &ackley,
        "DE",
        |f| DE::new(f, de_params.clone()).optimize(),
    ) {
        eprintln!("Error running DE on Ackley: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &ackley,
        "GA",
        |f| GA::new(f, ga_params.clone()).optimize(),
    ) {
        eprintln!("Error running GA on Ackley: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &ackley,
        "SA",
        |f| SA::new(f, sa_params.clone()).optimize(),
    ) {
        eprintln!("Error running SA on Ackley: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &ackley,
        "ACGWO",
        |f| Ok(ACGWO::new(f, acgwo_params.clone()).optimize()),
    ) {
        eprintln!("Error running ACGWO on Ackley: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &ackley,
        "ABCO",
        |f| ABCO::new(f, abco_params.clone()).optimize(),
    ) {
        eprintln!("Error running ABCO on Ackley: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &ackley,
        "GWO",
        |f| GWO::new(f, gwo_params.clone()).optimize(),
    ) {
        eprintln!("Error running GWO on Ackley: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &ackley,
        "FA",
        |f| {
            let (min_bounds, max_bounds) = f.bounds();
            let bounds: Vec<(f64, f64)> = min_bounds.into_iter()
                .zip(max_bounds.into_iter())
                .collect();
            let best_solution = fa.optimize(|x| f.evaluate(x), &bounds);
            let best_fitness = f.evaluate(&best_solution);
            Ok(OptimizationResult {
                best_solution,
                best_fitness,
                iterations: opt_params.max_iterations,
                evaluations: opt_params.max_iterations * opt_params.population_size,
            })
        },
    ) {
        eprintln!("Error running FA on Ackley: {}", e);
    }

    // Rosenbrock
    if let Err(e) = optimize_and_plot(
        &rosenbrock,
        "PSO",
        |f| PSO::new(f, pso_params.clone()).optimize(),
    ) {
        eprintln!("Error running PSO on Rosenbrock: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rosenbrock,
        "DE",
        |f| DE::new(f, de_params.clone()).optimize(),
    ) {
        eprintln!("Error running DE on Rosenbrock: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rosenbrock,
        "GA",
        |f| GA::new(f, ga_params.clone()).optimize(),
    ) {
        eprintln!("Error running GA on Rosenbrock: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rosenbrock,
        "SA",
        |f| SA::new(f, sa_params.clone()).optimize(),
    ) {
        eprintln!("Error running SA on Rosenbrock: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rosenbrock,
        "ACGWO",
        |f| Ok(ACGWO::new(f, acgwo_params.clone()).optimize()),
    ) {
        eprintln!("Error running ACGWO on Rosenbrock: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rosenbrock,
        "ABCO",
        |f| ABCO::new(f, abco_params.clone()).optimize(),
    ) {
        eprintln!("Error running ABCO on Rosenbrock: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rosenbrock,
        "GWO",
        |f| GWO::new(f, gwo_params.clone()).optimize(),
    ) {
        eprintln!("Error running GWO on Rosenbrock: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rosenbrock,
        "FA",
        |f| {
            let (min_bounds, max_bounds) = f.bounds();
            let bounds: Vec<(f64, f64)> = min_bounds.into_iter()
                .zip(max_bounds.into_iter())
                .collect();
            let best_solution = fa.optimize(|x| f.evaluate(x), &bounds);
            let best_fitness = f.evaluate(&best_solution);
            Ok(OptimizationResult {
                best_solution,
                best_fitness,
                iterations: opt_params.max_iterations,
                evaluations: opt_params.max_iterations * opt_params.population_size,
            })
        },
    ) {
        eprintln!("Error running FA on Rosenbrock: {}", e);
    }

    // Rastrigin
    if let Err(e) = optimize_and_plot(
        &rastrigin,
        "PSO",
        |f| PSO::new(f, pso_params.clone()).optimize(),
    ) {
        eprintln!("Error running PSO on Rastrigin: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rastrigin,
        "DE",
        |f| DE::new(f, de_params.clone()).optimize(),
    ) {
        eprintln!("Error running DE on Rastrigin: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rastrigin,
        "GA",
        |f| GA::new(f, ga_params.clone()).optimize(),
    ) {
        eprintln!("Error running GA on Rastrigin: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rastrigin,
        "SA",
        |f| SA::new(f, sa_params.clone()).optimize(),
    ) {
        eprintln!("Error running SA on Rastrigin: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rastrigin,
        "ACGWO",
        |f| Ok(ACGWO::new(f, acgwo_params.clone()).optimize()),
    ) {
        eprintln!("Error running ACGWO on Rastrigin: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rastrigin,
        "ABCO",
        |f| ABCO::new(f, abco_params.clone()).optimize(),
    ) {
        eprintln!("Error running ABCO on Rastrigin: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rastrigin,
        "GWO",
        |f| GWO::new(f, gwo_params.clone()).optimize(),
    ) {
        eprintln!("Error running GWO on Rastrigin: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &rastrigin,
        "FA",
        |f| {
            let (min_bounds, max_bounds) = f.bounds();
            let bounds: Vec<(f64, f64)> = min_bounds.into_iter()
                .zip(max_bounds.into_iter())
                .collect();
            let best_solution = fa.optimize(|x| f.evaluate(x), &bounds);
            let best_fitness = f.evaluate(&best_solution);
            Ok(OptimizationResult {
                best_solution,
                best_fitness,
                iterations: opt_params.max_iterations,
                evaluations: opt_params.max_iterations * opt_params.population_size,
            })
        },
    ) {
        eprintln!("Error running FA on Rastrigin: {}", e);
    }

    // Beale
    if let Err(e) = optimize_and_plot(
        &beale,
        "PSO",
        |f| PSO::new(f, pso_params.clone()).optimize(),
    ) {
        eprintln!("Error running PSO on Beale: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &beale,
        "DE",
        |f| DE::new(f, de_params.clone()).optimize(),
    ) {
        eprintln!("Error running DE on Beale: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &beale,
        "GA",
        |f| GA::new(f, ga_params.clone()).optimize(),
    ) {
        eprintln!("Error running GA on Beale: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &beale,
        "SA",
        |f| SA::new(f, sa_params.clone()).optimize(),
    ) {
        eprintln!("Error running SA on Beale: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &beale,
        "ACGWO",
        |f| Ok(ACGWO::new(f, acgwo_params.clone()).optimize()),
    ) {
        eprintln!("Error running ACGWO on Beale: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &beale,
        "ABCO",
        |f| ABCO::new(f, abco_params.clone()).optimize(),
    ) {
        eprintln!("Error running ABCO on Beale: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &beale,
        "GWO",
        |f| GWO::new(f, gwo_params.clone()).optimize(),
    ) {
        eprintln!("Error running GWO on Beale: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &beale,
        "FA",
        |f| {
            let (min_bounds, max_bounds) = f.bounds();
            let bounds: Vec<(f64, f64)> = min_bounds.into_iter()
                .zip(max_bounds.into_iter())
                .collect();
            let best_solution = fa.optimize(|x| f.evaluate(x), &bounds);
            let best_fitness = f.evaluate(&best_solution);
            Ok(OptimizationResult {
                best_solution,
                best_fitness,
                iterations: opt_params.max_iterations,
                evaluations: opt_params.max_iterations * opt_params.population_size,
            })
        },
    ) {
        eprintln!("Error running FA on Beale: {}", e);
    }

    // Griewank
    if let Err(e) = optimize_and_plot(
        &griewank,
        "PSO",
        |f| PSO::new(f, pso_params.clone()).optimize(),
    ) {
        eprintln!("Error running PSO on Griewank: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &griewank,
        "DE",
        |f| DE::new(f, de_params.clone()).optimize(),
    ) {
        eprintln!("Error running DE on Griewank: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &griewank,
        "GA",
        |f| GA::new(f, ga_params.clone()).optimize(),
    ) {
        eprintln!("Error running GA on Griewank: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &griewank,
        "SA",
        |f| SA::new(f, sa_params.clone()).optimize(),
    ) {
        eprintln!("Error running SA on Griewank: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &griewank,
        "ACGWO",
        |f| Ok(ACGWO::new(f, acgwo_params.clone()).optimize()),
    ) {
        eprintln!("Error running ACGWO on Griewank: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &griewank,
        "ABCO",
        |f| ABCO::new(f, abco_params.clone()).optimize(),
    ) {
        eprintln!("Error running ABCO on Griewank: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &griewank,
        "GWO",
        |f| GWO::new(f, gwo_params.clone()).optimize(),
    ) {
        eprintln!("Error running GWO on Griewank: {}", e);
    }
    if let Err(e) = optimize_and_plot(
        &griewank,
        "FA",
        |f| {
            let (min_bounds, max_bounds) = f.bounds();
            let bounds: Vec<(f64, f64)> = min_bounds.into_iter()
                .zip(max_bounds.into_iter())
                .collect();
            let best_solution = fa.optimize(|x| f.evaluate(x), &bounds);
            let best_fitness = f.evaluate(&best_solution);
            Ok(OptimizationResult {
                best_solution,
                best_fitness,
                iterations: opt_params.max_iterations,
                evaluations: opt_params.max_iterations * opt_params.population_size,
            })
        },
    ) {
        eprintln!("Error running FA on Griewank: {}", e);
    }

    Ok(())
}
