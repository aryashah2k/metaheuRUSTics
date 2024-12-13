use ndarray::{Array1, Array2};
use plotters::prelude::*;
use crate::algorithm::ObjectiveFunction;
use std::error::Error;

/// Plot a 2D optimization landscape as a contour plot
pub fn plot_2d_landscape(
    objective: &dyn ObjectiveFunction,
    result: Option<&Array1<f64>>,
    output_file: &str,
) -> Result<(), Box<dyn Error>> {
    // Only works for 2D functions
    assert_eq!(objective.dimensions(), 2, "Function must be 2D for visualization");
    
    let (lower_bounds, upper_bounds) = objective.bounds();
    let x_min = lower_bounds[0];
    let x_max = upper_bounds[0];
    let y_min = lower_bounds[1];
    let y_max = upper_bounds[1];
    
    // Create the plot
    let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Optimization Landscape (Contour)", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    
    chart.configure_mesh().draw()?;
    
    // Generate contour data
    let steps = 100;
    let x_step = (x_max - x_min) / steps as f64;
    let y_step = (y_max - y_min) / steps as f64;
    
    let mut data = vec![];
    for i in 0..=steps {
        let x = x_min + i as f64 * x_step;
        for j in 0..=steps {
            let y = y_min + j as f64 * y_step;
            let point = Array1::from_vec(vec![x, y]);
            let z = objective.evaluate(&point);
            data.push((x, y, z));
        }
    }
    
    // Draw contours
    let z_values: Vec<f64> = data.iter().map(|(_,_,z)| *z).collect();
    let z_min = z_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let z_max = z_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    let contours = 20;
    let z_step = (z_max - z_min) / contours as f64;
    
    for i in 0..contours {
        let z_level = z_min + i as f64 * z_step;
        let color = RGBColor(
            ((i as f64 / contours as f64) * 255.0) as u8,
            0,
            ((1.0 - i as f64 / contours as f64) * 255.0) as u8,
        );
        
        chart.draw_series(
            data.iter()
                .filter(|(_,_,z)| *z <= z_level + z_step && *z >= z_level)
                .map(|(x,y,_)| Circle::new((*x,*y), 1, color.filled())),
        )?;
    }
    
    // If a result point is provided, plot it
    if let Some(result_point) = result {
        chart.draw_series(std::iter::once(Circle::new(
            (result_point[0], result_point[1]),
            5,
            RED.filled(),
        )))?;
    }
    
    root.present()?;
    
    Ok(())
}

/// Plot a 2D optimization landscape as a 3D surface
pub fn plot_3d_landscape(
    objective: &dyn ObjectiveFunction,
    result: Option<&Array1<f64>>,
    output_file: &str,
) -> Result<(), Box<dyn Error>> {
    // Only works for 2D functions
    assert_eq!(objective.dimensions(), 2, "Function must be 2D for visualization");
    
    let (lower_bounds, upper_bounds) = objective.bounds();
    let x_min = lower_bounds[0];
    let x_max = upper_bounds[0];
    let y_min = lower_bounds[1];
    let y_max = upper_bounds[1];
    
    // Create the plot
    let root = BitMapBackend::new(output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Generate surface data
    let steps = 50;
    let x_step = (x_max - x_min) / steps as f64;
    let y_step = (y_max - y_min) / steps as f64;
    
    let mut data = vec![];
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    
    for i in 0..=steps {
        let x = x_min + i as f64 * x_step;
        for j in 0..=steps {
            let y = y_min + j as f64 * y_step;
            let point = Array1::from_vec(vec![x, y]);
            let z = objective.evaluate(&point);
            z_min = z_min.min(z);
            z_max = z_max.max(z);
            data.push((x, y, z));
        }
    }
    
    // Create 3D chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Optimization Landscape (3D)", ("sans-serif", 30))
        .margin(20)
        .build_cartesian_3d(x_min..x_max, z_min..z_max, y_min..y_max)?;
    
    chart.configure_axes()
        .light_grid_style(BLACK.mix(0.1))
        .max_light_lines(3)
        .draw()?;
    
    // Draw surface
    for i in 0..steps {
        for j in 0..steps {
            let idx1 = i * (steps + 1) + j;
            let idx2 = i * (steps + 1) + j + 1;
            let idx3 = (i + 1) * (steps + 1) + j;
            let idx4 = (i + 1) * (steps + 1) + j + 1;
            
            let points = [
                data[idx1],
                data[idx2],
                data[idx4],
                data[idx3],
            ];
            
            let z_avg = (points[0].2 + points[1].2 + points[2].2 + points[3].2) / 4.0;
            let color_scale = (z_avg - z_min) / (z_max - z_min);
            let color = RGBColor(
                ((1.0 - color_scale) * 255.0) as u8,
                0,
                (color_scale * 255.0) as u8,
            );
            
            chart.draw_series(std::iter::once(Polygon::new(
                points.iter().map(|&(x, y, z)| (x, z, y)).collect::<Vec<_>>(),
                &color.mix(0.8),
            )))?;
        }
    }
    
    // If a result point is provided, plot it
    if let Some(result_point) = result {
        let z = objective.evaluate(result_point);
        chart.draw_series(std::iter::once(Circle::new(
            (result_point[0], z, result_point[1]),
            5,
            RED.filled(),
        )))?;
    }
    
    root.present()?;
    
    Ok(())
}

/// Plot convergence history
pub fn plot_convergence(
    history: &[(usize, f64)],
    output_file: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let (iterations, values): (Vec<_>, Vec<_>) = history.iter().cloned().unzip();
    
    let min_value = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_value = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Convergence History", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0..*iterations.last().unwrap_or(&100),
            min_value..max_value,
        )?;
    
    chart.configure_mesh().draw()?;
    
    chart.draw_series(LineSeries::new(
        iterations.iter().zip(values.iter()).map(|(&x, &y)| (x, y)),
        &BLUE,
    ))?;
    
    root.present()?;
    
    Ok(())
}

/// Create an animation of the optimization process
pub struct OptimizationAnimator {
    frames: Vec<Array2<f64>>,
    objective: Box<dyn ObjectiveFunction>,
    output_dir: String,
}

impl OptimizationAnimator {
    pub fn new(objective: Box<dyn ObjectiveFunction>, output_dir: String) -> Self {
        Self {
            frames: Vec::new(),
            objective,
            output_dir,
        }
    }
    
    pub fn add_frame(&mut self, population: Array2<f64>) {
        self.frames.push(population);
    }
    
    pub fn save_animation(&self) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(&self.output_dir)?;
        
        for (i, frame) in self.frames.iter().enumerate() {
            let output_file = format!("{}/frame_{:04}.png", self.output_dir, i);
            
            // For 2D problems, we can visualize the population on the contour plot
            if self.objective.dimensions() == 2 {
                let root = BitMapBackend::new(&output_file, (800, 600)).into_drawing_area();
                root.fill(&WHITE)?;
                
                let (lower_bounds, upper_bounds) = self.objective.bounds();
                let mut chart = ChartBuilder::on(&root)
                    .caption(format!("Optimization Step {}", i), ("sans-serif", 30))
                    .margin(5)
                    .x_label_area_size(30)
                    .y_label_area_size(30)
                    .build_cartesian_2d(
                        lower_bounds[0]..upper_bounds[0],
                        lower_bounds[1]..upper_bounds[1],
                    )?;
                
                chart.configure_mesh().draw()?;
                
                // Draw population points
                chart.draw_series(
                    frame.rows()
                        .into_iter()
                        .map(|row| Circle::new((row[0], row[1]), 2, RED.filled())),
                )?;
                
                root.present()?;
            }
        }
        
        Ok(())
    }
}
