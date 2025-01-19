# MetaheuRUSTics

A comprehensive collection of metaheuristic optimization algorithms implemented in Rust. This library provides efficient implementations of popular optimization algorithms along with visualization tools.

## Features

- Multiple optimization algorithms:
  - Particle Swarm Optimization (PSO)
  - Differential Evolution (DE)
  - Genetic Algorithm (GA)
  - Simulated Annealing (SA)
  - Adaptive Chaotic Grey Wolf Optimizer (ACGWO)
  - Artificial Bee Colony Optimization (ABCO)
  - Grey Wolf Optimizer (GWO)
  - Firefly Algorithm (FA)

- Test functions for benchmarking:
  - Sphere Function
  - Ackley Function
  - Rosenbrock Function
  - Rastrigin Function
  - Beale Function
  - Griewank Function

- Visualization tools:
  - Surface plots
  - Contour plots
  - Convergence plots

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
metaheurustics = "0.2.0"
```

## Quick Start

Here's a simple example using PSO to optimize the Sphere function:

```rust
use metaheurustics::prelude::*;
use metaheurustics::algorithm::pso::{PSO, PSOParams};
use metaheurustics::test_function::Sphere;

fn main() {
    let sphere = Sphere::new();
    let pso_params = PSOParams::default();
    let pso = PSO::new(&sphere, pso_params);
    
    let result = pso.optimize();
    println!("Best solution: {:?}", result.best_solution);
    println!("Best fitness: {}", result.best_fitness);
}
```

## Examples

Check out the examples directory for more detailed usage:

- `examples/plot_all.rs`: Demonstrates visualization of all algorithms on different test functions
- `examples/simple_optimization.rs`: Shows basic usage of each optimization algorithm

## Documentation

For detailed documentation, visit [docs.rs/metaheurustics](https://docs.rs/metaheurustics-rs/0.2.0/metaheurustics/)

## Benchmarks

The library includes comprehensive benchmarks comparing the performance of different algorithms on various test functions. Run the benchmarks using:

```bash
cargo bench
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
