# MetaheuRUSTics

A comprehensive Rust library for metaheuristic optimization algorithms, providing efficient implementations of various optimization techniques.

[![Crates.io](https://img.shields.io/crates/v/metaheurustics.svg)](https://crates.io/crates/metaheurustics-rs)
[![Documentation](https://docs.rs/metaheurustics/badge.svg)](https://docs.rs/metaheurustics-rs/0.1.0/metaheurustics_rs/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Features

### Optimization Algorithms
- Particle Swarm Optimization (PSO)
- Differential Evolution (DE)
- Genetic Algorithm (GA)
- Simulated Annealing (SA)
- Artificial Bee Colony Optimization (ABCO)
- Grey Wolf Optimizer (GWO)
- Many more to come in future releases...

### Test Functions
- Sphere Function
- Ackley Function
- Rosenbrock Function
- Rastrigin Function
- Many more to come in future releases...

### Visualization
- 2D Contour Plots
- 3D Surface Plots
- Multiple Plot Formats (PNG)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
metaheurustics = "0.1.0"
```

## Quick Start

Here's a simple example using PSO to minimize the Sphere function:

```rust
use metaheurustics::prelude::*;

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
    
    // Create and run optimizer
    let optimizer = PSO::new(&sphere, params);
    let result = optimizer.optimize()?;
    
    println!("Best solution: {:?}", result.best_solution);
    println!("Best fitness: {}", result.best_fitness);
    
    Ok(())
}
```

## Documentation

For detailed documentation and examples, visit:
- [API Documentation](https://docs.rs/metaheurustics)
- [Examples Directory](https://github.com/aryashah2k/metaheurustics/tree/main/examples)

## Examples

The library includes several example programs:
1. `simple_optimization.rs`: Compare different algorithms on the Rosenbrock function
2. `optimize_sphere.rs`: Basic PSO optimization with visualization
3. `algorithm_comparison.rs`: Comprehensive comparison of all algorithms

Run examples using:
```bash
cargo run --example simple_optimization
cargo run --example optimize_sphere
cargo run --example algorithm_comparison
```

## Visualization

The library provides both 2D contour plots and 3D surface plots:
```rust
// Create plots directory
let plot_dir = "plots";
fs::create_dir_all(plot_dir)?;

// Generate plots
plot_2d_landscape(&problem, Some(&result.best_solution), "contour.png")?;
plot_3d_landscape(&problem, Some(&result.best_solution), "surface.png")?;
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:
```bibtex
@software{metaheurustics,
  author = {Shah, Arya},
  title = {MetaheuRUSTics: A Rust Library for Optimization Algorithms},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/aryashah2k/metaheurustics}
}
