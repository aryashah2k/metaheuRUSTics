use metaheurustics::algorithm::{ACGWO, ACGWOParams, FireflyAlgorithm, OptimizationParams, ObjectiveFunction};
use metaheurustics::test_function::{Beale, Sphere, TestFunction};
use plotters::prelude::*;
use ndarray::Array1;

fn plot_optimization_surface(
    test_function: &dyn TestFunction,
    best_solution: &Array1<f64>,
    best_fitness: f64,
    algorithm_name: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create directory for plots if it doesn't exist
    std::fs::create_dir_all("plots")?;

    // Generate surface data
    let (x, y, z) = test_function.test_3d_surface(100);
    let (min_bounds, max_bounds) = test_function.bounds();

    // Create visualization
    let root = BitMapBackend::new(filename, (800, 600))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let x_range = (min_bounds[0]..max_bounds[0]).step(0.1);
    let y_range = (min_bounds[1]..max_bounds[1]).step(0.1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} Optimization using {}", test_function.name(), algorithm_name),
            ("sans-serif", 30),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    // Plot contour lines
    let z_min = z.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let z_max = z.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let levels = (0..10).map(|i| {
        z_min + (z_max - z_min) * (i as f64 / 10.0)
    }).collect::<Vec<_>>();

    for level in levels {
        let mut points = Vec::new();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                if (z[[i, j]] - level).abs() < 0.1 {
                    points.push((x[[i, j]], y[[i, j]]));
                }
            }
        }

        chart.draw_series(LineSeries::new(
            points,
            &BLUE.mix(0.3),
        ))?;
    }

    // Plot the optimal point
    chart.draw_series(PointSeries::of_element(
        vec![(best_solution[0], best_solution[1])],
        5,
        &RED,
        &|c, s, st| {
            return EmptyElement::at(c)
                + Circle::new((0, 0), s, st.filled())
                + Text::new(
                    format!("Optimal Point ({:.3}, {:.3})\nFitness: {:.6}", c.0, c.1, best_fitness),
                    (10, 0),
                    ("sans-serif", 15),
                );
        },
    ))?;

    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Plot has been saved to '{}'", filename);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test functions
    let sphere = Sphere::new(2);
    let beale = Beale::new();

    // Configure ACGWO parameters
    let acgwo_params = ACGWOParams {
        opt_params: OptimizationParams {
            population_size: 50,
            max_iterations: 300,
            target_value: Some(1e-6),
        },
        lambda: 0.7,
        a_decay: 0.005,
    };

    // Configure Firefly Algorithm parameters
    let fa = FireflyAlgorithm::new(50, 300);

    // Optimize and plot Sphere function with ACGWO
    let acgwo_sphere = ACGWO::new(&sphere, acgwo_params.clone());
    let acgwo_sphere_result = acgwo_sphere.optimize();
    plot_optimization_surface(
        &sphere,
        &acgwo_sphere_result.best_solution,
        acgwo_sphere_result.best_fitness,
        "ACGWO",
        "plots/sphere_acgwo.png",
    )?;

    // Convert bounds for FA
    let (sphere_min, sphere_max) = sphere.bounds();
    let sphere_bounds: Vec<(f64, f64)> = sphere_min.into_iter()
        .zip(sphere_max.into_iter())
        .collect();

    // Optimize and plot Sphere function with FA
    let fa_sphere_result = fa.optimize(|x| sphere.evaluate(x), &sphere_bounds);
    let fa_sphere_fitness = sphere.evaluate(&fa_sphere_result);
    plot_optimization_surface(
        &sphere,
        &fa_sphere_result,
        fa_sphere_fitness,
        "Firefly Algorithm",
        "plots/sphere_fa.png",
    )?;

    // Optimize and plot Beale function with ACGWO
    let acgwo_beale = ACGWO::new(&beale, acgwo_params);
    let acgwo_beale_result = acgwo_beale.optimize();
    plot_optimization_surface(
        &beale,
        &acgwo_beale_result.best_solution,
        acgwo_beale_result.best_fitness,
        "ACGWO",
        "plots/beale_acgwo.png",
    )?;

    // Convert bounds for FA
    let (beale_min, beale_max) = beale.bounds();
    let beale_bounds: Vec<(f64, f64)> = beale_min.into_iter()
        .zip(beale_max.into_iter())
        .collect();

    // Optimize and plot Beale function with FA
    let fa_beale_result = fa.optimize(|x| beale.evaluate(x), &beale_bounds);
    let fa_beale_fitness = beale.evaluate(&fa_beale_result);
    plot_optimization_surface(
        &beale,
        &fa_beale_result,
        fa_beale_fitness,
        "Firefly Algorithm",
        "plots/beale_fa.png",
    )?;

    Ok(())
}
