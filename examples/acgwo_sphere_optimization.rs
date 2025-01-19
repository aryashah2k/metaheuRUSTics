use metaheurustics::algorithm::{ACGWO, ACGWOParams, OptimizationParams, ObjectiveFunction};
use metaheurustics::test_function::Sphere;
use plotters::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the Sphere test function
    let sphere = Sphere::new(2);
    
    // Configure ACGWO parameters
    let params = ACGWOParams {
        opt_params: OptimizationParams {
            population_size: 50,
            max_iterations: 300,
            target_value: Some(1e-6),
        },
        lambda: 0.7,
        a_decay: 0.005,
    };

    // Create and run optimizer
    let optimizer = ACGWO::new(&sphere, params);
    let result = optimizer.optimize();

    // Print results
    println!("Optimization Results:");
    println!("Best Solution: {:?}", result.best_solution);
    println!("Best Fitness: {}", result.best_fitness);
    println!("Total Iterations: {}", result.iterations);
    println!("Total Function Evaluations: {}", result.evaluations);

    // Create directory for plots if it doesn't exist
    std::fs::create_dir_all("plots")?;

    // Create visualization
    let root = BitMapBackend::new("plots/sphere_acgwo.png", (800, 600))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let (min_bounds, max_bounds) = sphere.bounds();
    let x_range = (min_bounds[0]..max_bounds[0]).step(0.1);
    let y_range = (min_bounds[1]..max_bounds[1]).step(0.1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Sphere Function Optimization using ACGWO", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    // Plot the optimal point
    chart.draw_series(PointSeries::of_element(
        vec![(result.best_solution[0], result.best_solution[1])],
        5,
        &RED,
        &|c, s, st| {
            return EmptyElement::at(c)
                + Circle::new((0, 0), s, st.filled())
                + Text::new(
                    format!("Optimal Point ({:.3}, {:.3})", c.0, c.1),
                    (10, 0),
                    ("sans-serif", 15),
                );
        },
    ))?;

    // Add contour plot
    let n_points = 100;
    let x_step = (max_bounds[0] - min_bounds[0]) / n_points as f64;
    let y_step = (max_bounds[1] - min_bounds[1]) / n_points as f64;

    let mut contours = Vec::new();
    for i in 0..n_points {
        for j in 0..n_points {
            let x = min_bounds[0] + i as f64 * x_step;
            let y = min_bounds[1] + j as f64 * y_step;
            let point = Array1::from_vec(vec![x, y]);
            let z = sphere.evaluate(&point);
            contours.push((x, y, z));
        }
    }

    // Create contour lines
    let levels = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    for level in levels {
        let points: Vec<(f64, f64)> = contours.iter()
            .filter(|(_, _, z)| (*z - level).abs() < 0.1)
            .map(|(x, y, _)| (*x, *y))
            .collect();

        chart.draw_series(LineSeries::new(
            points,
            &BLUE.mix(0.3),
        ))?;
    }

    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Plot has been saved to 'plots/sphere_acgwo.png'");

    Ok(())
}
