use criterion::{criterion_group, criterion_main, Criterion};
use metaheurustics::algorithm::{
    OptimizationParams,
    ObjectiveFunction,
    pso::{PSO, PSOParams},
    de::{DE, DEParams, DEStrategy},
    ga::{GA, GAParams},
    sa::{SA, SAParams},
    abco::{ABCO, ABCOParams},
    gwo::{GWO, GWOParams},
};
use metaheurustics::test_function::{
    Sphere,
    Ackley,
    Rosenbrock,
    Rastrigin,
    Griewank,
    Schwefel,
    Levy,
    Michalewicz,
};

fn create_test_functions(dim: usize) -> Vec<Box<dyn ObjectiveFunction>> {
    vec![
        Box::new(Sphere::new(dim)),
        Box::new(Ackley::new(dim)),
        Box::new(Rosenbrock::new(dim)),
        Box::new(Rastrigin::new(dim)),
        Box::new(Griewank::new(dim)),
        Box::new(Schwefel::new(dim)),
        Box::new(Levy::new(dim)),
        Box::new(Michalewicz::new(dim, 10.0)),
    ]
}

fn benchmark_pso(c: &mut Criterion) {
    let dim = 30;
    let test_functions = create_test_functions(dim);
    
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

    for (i, func) in test_functions.iter().enumerate() {
        c.bench_function(&format!("PSO_func_{}", i), |b| {
            b.iter(|| {
                let optimizer = PSO::new(func.as_ref(), params.clone());
                optimizer.optimize()
            })
        });
    }
}

fn benchmark_de(c: &mut Criterion) {
    let dim = 30;
    let test_functions = create_test_functions(dim);
    
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

    for (i, func) in test_functions.iter().enumerate() {
        c.bench_function(&format!("DE_func_{}", i), |b| {
            b.iter(|| {
                let optimizer = DE::new(func.as_ref(), params.clone());
                optimizer.optimize()
            })
        });
    }
}

fn benchmark_ga(c: &mut Criterion) {
    let dim = 30;
    let test_functions = create_test_functions(dim);
    
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

    for (i, func) in test_functions.iter().enumerate() {
        c.bench_function(&format!("GA_func_{}", i), |b| {
            b.iter(|| {
                let optimizer = GA::new(func.as_ref(), params.clone());
                optimizer.optimize()
            })
        });
    }
}

fn benchmark_sa(c: &mut Criterion) {
    let dim = 30;
    let test_functions = create_test_functions(dim);
    
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

    for (i, func) in test_functions.iter().enumerate() {
        c.bench_function(&format!("SA_func_{}", i), |b| {
            b.iter(|| {
                let optimizer = SA::new(func.as_ref(), params.clone());
                optimizer.optimize()
            })
        });
    }
}

fn benchmark_abco(c: &mut Criterion) {
    let dim = 30;
    let test_functions = create_test_functions(dim);
    
    let params = ABCOParams {
        opt_params: OptimizationParams {
            population_size: 30,
            max_iterations: 100,
            target_value: None,
        },
        limit: 20,
    };

    for (i, func) in test_functions.iter().enumerate() {
        c.bench_function(&format!("ABCO_func_{}", i), |b| {
            b.iter(|| {
                let optimizer = ABCO::new(func.as_ref(), params.clone());
                optimizer.optimize()
            })
        });
    }
}

fn benchmark_gwo(c: &mut Criterion) {
    let dim = 30;
    let test_functions = create_test_functions(dim);
    
    let params = GWOParams {
        opt_params: OptimizationParams {
            population_size: 30,
            max_iterations: 100,
            target_value: None,
        },
        a_decay: 0.01,
    };

    for (i, func) in test_functions.iter().enumerate() {
        c.bench_function(&format!("GWO_func_{}", i), |b| {
            b.iter(|| {
                let optimizer = GWO::new(func.as_ref(), params.clone());
                optimizer.optimize()
            })
        });
    }
}

criterion_group!(
    benches,
    benchmark_pso,
    benchmark_de,
    benchmark_ga,
    benchmark_sa,
    benchmark_abco,
    benchmark_gwo,
);
criterion_main!(benches);
