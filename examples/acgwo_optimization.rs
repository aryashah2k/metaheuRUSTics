use metaheurustics::algorithm::{ACGWO, ACGWOParams, OptimizationParams, ObjectiveFunction};
use metaheurustics::test_function::Beale;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the Beale test function
    let beale = Beale::new();
    
    // Configure ACGWO parameters
    let params = ACGWOParams {
        opt_params: OptimizationParams {
            population_size: 50,
            max_iterations: 200,
            target_value: Some(1e-6),
        },
        lambda: 0.5,
        a_decay: 0.01,
    };

    // Create and run optimizer
    let optimizer = ACGWO::new(&beale, params);
    let result = optimizer.optimize();

    // Print results
    println!("Optimization Results:");
    println!("Best Solution: {:?}", result.best_solution);
    println!("Best Fitness: {}", result.best_fitness);
    println!("Total Iterations: {}", result.iterations);
    println!("Total Function Evaluations: {}", result.evaluations);

    // Create visualization
    let root = BitMapBackend::new("plots/beale_acgwo.png", (800, 600))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let (min_bounds, max_bounds) = beale.bounds();
    let x_range = (min_bounds[0]..max_bounds[0]).step(0.1);
    let y_range = (min_bounds[1]..max_bounds[1]).step(0.1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Beale Function Optimization using ACGWO", ("sans-serif", 30))
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

    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Plot has been saved to 'plots/beale_acgwo.png'");

    Ok(())
}
