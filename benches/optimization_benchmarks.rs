use criterion::{criterion_group, criterion_main, Criterion};
use metaheurustics::algorithm::{
    ObjectiveFunction,
    OptimizationParams,
    pso::{PSO, PSOParams},
    de::{DE, DEParams, DEStrategy},
    ga::{GA, GAParams},
    sa::{SA, SAParams},
    acgwo::{ACGWO, ACGWOParams},
    abco::{ABCO, ABCOParams},
    gwo::{GWO, GWOParams},
    fa::{FireflyAlgorithm, FireflyParams},
};
use metaheurustics::test_function::{
    Sphere,
    Ackley,
    Rosenbrock,
    Rastrigin,
    Beale,
    Griewank,
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
    
    let griewank = Griewank::new(dim);
    c.bench_function("griewank", |b| {
        b.iter(|| griewank.evaluate(&x))
    });
}

fn benchmark_all_algorithms(c: &mut Criterion) {
    let dims = vec![2, 10, 30];
    let opt_params = OptimizationParams {
        population_size: 50,
        max_iterations: 300,
        target_value: Some(1e-6),
    };

    for dim in dims {
        // Sphere
        {
            let func = Sphere::new(dim);
            let func_name = "sphere";
            
            // PSO
            let pso_params = PSOParams {
                opt_params: opt_params.clone(),
                c1: 2.0,
                c2: 2.0,
                w: 0.7,
                w_decay: 0.99,
            };
            c.bench_function(&format!("pso_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| PSO::new(&func, pso_params.clone()).optimize())
            });

            // DE
            let de_params = DEParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.9,
                mutation_factor: 0.8,
                strategy: DEStrategy::RandOneBin,
            };
            c.bench_function(&format!("de_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| DE::new(&func, de_params.clone()).optimize())
            });

            // GA
            let ga_params = GAParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.8,
                mutation_rate: 0.1,
                tournament_size: 3,
            };
            c.bench_function(&format!("ga_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GA::new(&func, ga_params.clone()).optimize())
            });

            // SA
            let sa_params = SAParams {
                opt_params: opt_params.clone(),
                initial_temp: 1.0,
                cooling_rate: 0.95,
                iterations_per_temp: 50,
                min_temp: 1e-10,
            };
            c.bench_function(&format!("sa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| SA::new(&func, sa_params.clone()).optimize())
            });

            // ACGWO
            let acgwo_params = ACGWOParams {
                opt_params: opt_params.clone(),
                lambda: 0.7,
                a_decay: 0.005,
            };
            c.bench_function(&format!("acgwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ACGWO::new(&func, acgwo_params.clone()).optimize())
            });

            // ABCO
            let abco_params = ABCOParams {
                opt_params: opt_params.clone(),
                limit: 20,
            };
            c.bench_function(&format!("abco_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ABCO::new(&func, abco_params.clone()).optimize())
            });

            // GWO
            let gwo_params = GWOParams {
                opt_params: opt_params.clone(),
                a_decay: 0.01,
            };
            c.bench_function(&format!("gwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GWO::new(&func, gwo_params.clone()).optimize())
            });

            // FA
            let fa_params = FireflyParams {
                opt_params: opt_params.clone(),
                alpha: 0.5,
                beta: 0.2,
                gamma: 1.0,
            };
            c.bench_function(&format!("fa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| FireflyAlgorithm::new(&func, fa_params.clone()).optimize())
            });
        }

        // Ackley
        {
            let func = Ackley::new(dim);
            let func_name = "ackley";
            
            // PSO
            let pso_params = PSOParams {
                opt_params: opt_params.clone(),
                c1: 2.0,
                c2: 2.0,
                w: 0.7,
                w_decay: 0.99,
            };
            c.bench_function(&format!("pso_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| PSO::new(&func, pso_params.clone()).optimize())
            });

            // DE
            let de_params = DEParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.9,
                mutation_factor: 0.8,
                strategy: DEStrategy::RandOneBin,
            };
            c.bench_function(&format!("de_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| DE::new(&func, de_params.clone()).optimize())
            });

            // GA
            let ga_params = GAParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.8,
                mutation_rate: 0.1,
                tournament_size: 3,
            };
            c.bench_function(&format!("ga_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GA::new(&func, ga_params.clone()).optimize())
            });

            // SA
            let sa_params = SAParams {
                opt_params: opt_params.clone(),
                initial_temp: 1.0,
                cooling_rate: 0.95,
                iterations_per_temp: 50,
                min_temp: 1e-10,
            };
            c.bench_function(&format!("sa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| SA::new(&func, sa_params.clone()).optimize())
            });

            // ACGWO
            let acgwo_params = ACGWOParams {
                opt_params: opt_params.clone(),
                lambda: 0.7,
                a_decay: 0.005,
            };
            c.bench_function(&format!("acgwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ACGWO::new(&func, acgwo_params.clone()).optimize())
            });

            // ABCO
            let abco_params = ABCOParams {
                opt_params: opt_params.clone(),
                limit: 20,
            };
            c.bench_function(&format!("abco_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ABCO::new(&func, abco_params.clone()).optimize())
            });

            // GWO
            let gwo_params = GWOParams {
                opt_params: opt_params.clone(),
                a_decay: 0.01,
            };
            c.bench_function(&format!("gwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GWO::new(&func, gwo_params.clone()).optimize())
            });

            // FA
            let fa_params = FireflyParams {
                opt_params: opt_params.clone(),
                alpha: 0.5,
                beta: 0.2,
                gamma: 1.0,
            };
            c.bench_function(&format!("fa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| FireflyAlgorithm::new(&func, fa_params.clone()).optimize())
            });
        }

        // Rosenbrock
        {
            let func = Rosenbrock::new(dim);
            let func_name = "rosenbrock";
            
            // PSO
            let pso_params = PSOParams {
                opt_params: opt_params.clone(),
                c1: 2.0,
                c2: 2.0,
                w: 0.7,
                w_decay: 0.99,
            };
            c.bench_function(&format!("pso_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| PSO::new(&func, pso_params.clone()).optimize())
            });

            // DE
            let de_params = DEParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.9,
                mutation_factor: 0.8,
                strategy: DEStrategy::RandOneBin,
            };
            c.bench_function(&format!("de_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| DE::new(&func, de_params.clone()).optimize())
            });

            // GA
            let ga_params = GAParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.8,
                mutation_rate: 0.1,
                tournament_size: 3,
            };
            c.bench_function(&format!("ga_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GA::new(&func, ga_params.clone()).optimize())
            });

            // SA
            let sa_params = SAParams {
                opt_params: opt_params.clone(),
                initial_temp: 1.0,
                cooling_rate: 0.95,
                iterations_per_temp: 50,
                min_temp: 1e-10,
            };
            c.bench_function(&format!("sa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| SA::new(&func, sa_params.clone()).optimize())
            });

            // ACGWO
            let acgwo_params = ACGWOParams {
                opt_params: opt_params.clone(),
                lambda: 0.7,
                a_decay: 0.005,
            };
            c.bench_function(&format!("acgwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ACGWO::new(&func, acgwo_params.clone()).optimize())
            });

            // ABCO
            let abco_params = ABCOParams {
                opt_params: opt_params.clone(),
                limit: 20,
            };
            c.bench_function(&format!("abco_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ABCO::new(&func, abco_params.clone()).optimize())
            });

            // GWO
            let gwo_params = GWOParams {
                opt_params: opt_params.clone(),
                a_decay: 0.01,
            };
            c.bench_function(&format!("gwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GWO::new(&func, gwo_params.clone()).optimize())
            });

            // FA
            let fa_params = FireflyParams {
                opt_params: opt_params.clone(),
                alpha: 0.5,
                beta: 0.2,
                gamma: 1.0,
            };
            c.bench_function(&format!("fa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| FireflyAlgorithm::new(&func, fa_params.clone()).optimize())
            });
        }

        // Rastrigin
        {
            let func = Rastrigin::new(dim);
            let func_name = "rastrigin";
            
            // PSO
            let pso_params = PSOParams {
                opt_params: opt_params.clone(),
                c1: 2.0,
                c2: 2.0,
                w: 0.7,
                w_decay: 0.99,
            };
            c.bench_function(&format!("pso_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| PSO::new(&func, pso_params.clone()).optimize())
            });

            // DE
            let de_params = DEParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.9,
                mutation_factor: 0.8,
                strategy: DEStrategy::RandOneBin,
            };
            c.bench_function(&format!("de_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| DE::new(&func, de_params.clone()).optimize())
            });

            // GA
            let ga_params = GAParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.8,
                mutation_rate: 0.1,
                tournament_size: 3,
            };
            c.bench_function(&format!("ga_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GA::new(&func, ga_params.clone()).optimize())
            });

            // SA
            let sa_params = SAParams {
                opt_params: opt_params.clone(),
                initial_temp: 1.0,
                cooling_rate: 0.95,
                iterations_per_temp: 50,
                min_temp: 1e-10,
            };
            c.bench_function(&format!("sa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| SA::new(&func, sa_params.clone()).optimize())
            });

            // ACGWO
            let acgwo_params = ACGWOParams {
                opt_params: opt_params.clone(),
                lambda: 0.7,
                a_decay: 0.005,
            };
            c.bench_function(&format!("acgwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ACGWO::new(&func, acgwo_params.clone()).optimize())
            });

            // ABCO
            let abco_params = ABCOParams {
                opt_params: opt_params.clone(),
                limit: 20,
            };
            c.bench_function(&format!("abco_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ABCO::new(&func, abco_params.clone()).optimize())
            });

            // GWO
            let gwo_params = GWOParams {
                opt_params: opt_params.clone(),
                a_decay: 0.01,
            };
            c.bench_function(&format!("gwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GWO::new(&func, gwo_params.clone()).optimize())
            });

            // FA
            let fa_params = FireflyParams {
                opt_params: opt_params.clone(),
                alpha: 0.5,
                beta: 0.2,
                gamma: 1.0,
            };
            c.bench_function(&format!("fa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| FireflyAlgorithm::new(&func, fa_params.clone()).optimize())
            });
        }

        // Griewank
        {
            let func = Griewank::new(dim);
            let func_name = "griewank";
            
            // PSO
            let pso_params = PSOParams {
                opt_params: opt_params.clone(),
                c1: 2.0,
                c2: 2.0,
                w: 0.7,
                w_decay: 0.99,
            };
            c.bench_function(&format!("pso_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| PSO::new(&func, pso_params.clone()).optimize())
            });

            // DE
            let de_params = DEParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.9,
                mutation_factor: 0.8,
                strategy: DEStrategy::RandOneBin,
            };
            c.bench_function(&format!("de_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| DE::new(&func, de_params.clone()).optimize())
            });

            // GA
            let ga_params = GAParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.8,
                mutation_rate: 0.1,
                tournament_size: 3,
            };
            c.bench_function(&format!("ga_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GA::new(&func, ga_params.clone()).optimize())
            });

            // SA
            let sa_params = SAParams {
                opt_params: opt_params.clone(),
                initial_temp: 1.0,
                cooling_rate: 0.95,
                iterations_per_temp: 50,
                min_temp: 1e-10,
            };
            c.bench_function(&format!("sa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| SA::new(&func, sa_params.clone()).optimize())
            });

            // ACGWO
            let acgwo_params = ACGWOParams {
                opt_params: opt_params.clone(),
                lambda: 0.7,
                a_decay: 0.005,
            };
            c.bench_function(&format!("acgwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ACGWO::new(&func, acgwo_params.clone()).optimize())
            });

            // ABCO
            let abco_params = ABCOParams {
                opt_params: opt_params.clone(),
                limit: 20,
            };
            c.bench_function(&format!("abco_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ABCO::new(&func, abco_params.clone()).optimize())
            });

            // GWO
            let gwo_params = GWOParams {
                opt_params: opt_params.clone(),
                a_decay: 0.01,
            };
            c.bench_function(&format!("gwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GWO::new(&func, gwo_params.clone()).optimize())
            });

            // FA
            let fa_params = FireflyParams {
                opt_params: opt_params.clone(),
                alpha: 0.5,
                beta: 0.2,
                gamma: 1.0,
            };
            c.bench_function(&format!("fa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| FireflyAlgorithm::new(&func, fa_params.clone()).optimize())
            });
        }

        // Beale
        {
            let func = Beale::new();
            let func_name = "beale";
            
            // PSO
            let pso_params = PSOParams {
                opt_params: opt_params.clone(),
                c1: 2.0,
                c2: 2.0,
                w: 0.7,
                w_decay: 0.99,
            };
            c.bench_function(&format!("pso_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| PSO::new(&func, pso_params.clone()).optimize())
            });

            // DE
            let de_params = DEParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.9,
                mutation_factor: 0.8,
                strategy: DEStrategy::RandOneBin,
            };
            c.bench_function(&format!("de_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| DE::new(&func, de_params.clone()).optimize())
            });

            // GA
            let ga_params = GAParams {
                opt_params: opt_params.clone(),
                crossover_rate: 0.8,
                mutation_rate: 0.1,
                tournament_size: 3,
            };
            c.bench_function(&format!("ga_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GA::new(&func, ga_params.clone()).optimize())
            });

            // SA
            let sa_params = SAParams {
                opt_params: opt_params.clone(),
                initial_temp: 1.0,
                cooling_rate: 0.95,
                iterations_per_temp: 50,
                min_temp: 1e-10,
            };
            c.bench_function(&format!("sa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| SA::new(&func, sa_params.clone()).optimize())
            });

            // ACGWO
            let acgwo_params = ACGWOParams {
                opt_params: opt_params.clone(),
                lambda: 0.7,
                a_decay: 0.005,
            };
            c.bench_function(&format!("acgwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ACGWO::new(&func, acgwo_params.clone()).optimize())
            });

            // ABCO
            let abco_params = ABCOParams {
                opt_params: opt_params.clone(),
                limit: 20,
            };
            c.bench_function(&format!("abco_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| ABCO::new(&func, abco_params.clone()).optimize())
            });

            // GWO
            let gwo_params = GWOParams {
                opt_params: opt_params.clone(),
                a_decay: 0.01,
            };
            c.bench_function(&format!("gwo_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| GWO::new(&func, gwo_params.clone()).optimize())
            });

            // FA
            let fa_params = FireflyParams {
                opt_params: opt_params.clone(),
                alpha: 0.5,
                beta: 0.2,
                gamma: 1.0,
            };
            c.bench_function(&format!("fa_{}_{}_dim", func_name, dim), |b| {
                b.iter(|| FireflyAlgorithm::new(&func, fa_params.clone()).optimize())
            });
        }
    }
}

criterion_group!(
    benches,
    benchmark_functions,
    benchmark_all_algorithms,
);
criterion_main!(benches);
