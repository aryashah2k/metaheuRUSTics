use criterion::{criterion_group, criterion_main, Criterion};
use metaheurustics::algorithm::{
    ObjectiveFunction,
    OptimizationParams,
    pso::{PSO, PSOParams},
    de::{DE, DEParams, DEStrategy},
    ga::{GA, GAParams},
    sa::{SA, SAParams},
};
use metaheurustics::test_function::{
    Sphere,
    Ackley,
    Rosenbrock,
    Rastrigin,
};
use ndarray::Array1;

fn benchmark_functions(c: &mut Criterion) {
    let dim = 30;
    let x = Array1::zeros(dim);
    
    let sphere = Sphere::new(dim);
    c.bench_function("sphere", |b| {
        b.iter(|| sphere.evaluate(&x))
    });
    
    let ackley = Ackley::new(dim);
    c.bench_function("ackley", |b| {
        b.iter(|| ackley.evaluate(&x))
    });
    
    let rosenbrock = Rosenbrock::new(dim);
    c.bench_function("rosenbrock", |b| {
        b.iter(|| rosenbrock.evaluate(&x))
    });
    
    let rastrigin = Rastrigin::new(dim);
    c.bench_function("rastrigin", |b| {
        b.iter(|| rastrigin.evaluate(&x))
    });
}

fn benchmark_pso(c: &mut Criterion) {
    let dim = 30;
    let problem: Box<dyn ObjectiveFunction> = Box::new(Sphere::new(dim));
    
    let params = PSOParams {
        opt_params: OptimizationParams {
            population_size: 30,
            max_iterations: 100,
            target_value: None,
        },
        w: 0.7,
        c1: 2.0,
        c2: 2.0,
        w_decay: 0.99,
    };

    c.bench_function("PSO", |b| {
        b.iter(|| {
            let optimizer = PSO::new(problem.as_ref(), params.clone());
            optimizer.optimize()
        })
    });
}

fn benchmark_de(c: &mut Criterion) {
    let dim = 30;
    let problem: Box<dyn ObjectiveFunction> = Box::new(Sphere::new(dim));
    
    let params = DEParams {
        opt_params: OptimizationParams {
            population_size: 30,
            max_iterations: 100,
            target_value: None,
        },
        strategy: DEStrategy::RandOneBin,
        crossover_rate: 0.9,
        mutation_factor: 0.8,
    };

    c.bench_function("DE", |b| {
        b.iter(|| {
            let optimizer = DE::new(problem.as_ref(), params.clone());
            optimizer.optimize()
        })
    });
}

fn benchmark_ga(c: &mut Criterion) {
    let dim = 30;
    let problem: Box<dyn ObjectiveFunction> = Box::new(Sphere::new(dim));
    
    let params = GAParams {
        opt_params: OptimizationParams {
            population_size: 30,
            max_iterations: 100,
            target_value: None,
        },
        crossover_rate: 0.8,
        mutation_rate: 0.1,
        tournament_size: 3,
    };

    c.bench_function("GA", |b| {
        b.iter(|| {
            let optimizer = GA::new(problem.as_ref(), params.clone());
            optimizer.optimize()
        })
    });
}

fn benchmark_sa(c: &mut Criterion) {
    let dim = 30;
    let problem: Box<dyn ObjectiveFunction> = Box::new(Sphere::new(dim));
    
    let params = SAParams {
        opt_params: OptimizationParams {
            population_size: 1,
            max_iterations: 1000,
            target_value: None,
        },
        initial_temp: 100.0,
        cooling_rate: 0.95,
        iterations_per_temp: 50,
        min_temp: 1e-10,
    };

    c.bench_function("SA", |b| {
        b.iter(|| {
            let optimizer = SA::new(problem.as_ref(), params.clone());
            optimizer.optimize()
        })
    });
}

criterion_group!(
    benches,
    benchmark_functions,
    benchmark_pso,
    benchmark_de,
    benchmark_ga,
    benchmark_sa
);
criterion_main!(benches);
