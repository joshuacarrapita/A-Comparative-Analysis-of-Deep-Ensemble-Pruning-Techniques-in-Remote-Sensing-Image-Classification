# system imports
import argparse
import sys
import time
import os

# 3rd party imports
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pathlib import Path

# local imports
from epfd.data_distributed import (distributed_single_pruning,
                                   distributed_pruning_methods,
                                   COMEP_Pruning, DOMEP_Pruning)
import gasen.GA as GA
from PyPruning.Papers import create_pruner

# --- GLOBAL DEFINITIONS ---
EPFD_ALGOS = [
    'ES', 'KP', 'KL', 'RE', 'OO', 'DREP', 'SEP', 'OEP', 'PEP', 
    'GMA', 'LCS', 'COMEP', 'DOMEP'
]
GASEN_ALGOS = ['GASEN']
PYPRUNING_ALGOS = [
    "individual_margin_diversity", "individual_contribution", "individual_error",
    "individual_kappa_statistic", "reduced_error", "complementariness", "drep",
    "margin_distance", "combined", "reference_vector", "combined_error",
    "error_ambiguity", "largest_mean_distance", "cluster_accuracy", 
    "cluster_centroids", "random"
]

ALL_ALGOS = EPFD_ALGOS + GASEN_ALGOS + PYPRUNING_ALGOS

# --- HELPER FUNCTIONS ---
def load_data(results_path, targets_path):
    """Loads tensors and infers dimensions M, N, C."""
    print(f"\nLoading data from:\n  Probas: {results_path}\n  Targets: {targets_path}")
    
    probas = torch.load(Path(results_path).resolve()).float().numpy()
    targets = torch.load(Path(targets_path).resolve()).float().numpy()
    
    M, N, C = probas.shape

    print(f"Inferred Dimensions -> Models (M): {M}, Samples (N): {N}, Classes (C): {C}")
    return probas, targets, M, N, C

# --- ALGORITHM IMPLEMENTATIONS ---
def run_epfd_algo(algo_name, probas, targets, nb_pru, args):
    """Wrapper for EPFD library logic."""

    M, N, C = probas.shape
    
    # Get Hard Labels
    # Input is (M, N, C). argmax over classes (axis 2) -> (M, N)
    results_hard = np.argmax(probas, axis=2) 
    
    rho = nb_pru / M
    selected_indices = []

    start_time = time.time()
    
    if algo_name in ['COMEP', 'DOMEP']:
        results_transposed_list = results_hard.T.tolist()
        if algo_name == 'COMEP':
            selected_indices = COMEP_Pruning(results_transposed_list, nb_pru, targets, args.lam)
        elif algo_name == 'DOMEP':
            selected_indices = DOMEP_Pruning(results_transposed_list, nb_pru, args.m, targets, args.lam)
    else:
        if args.distributed:
            selected_indices = distributed_pruning_methods(targets, results_hard, nb_pru, args.m, algo_name, rho=rho)
        else:
            selected_indices = distributed_single_pruning(algo_name, targets, results_hard, M, nb_pru, rho=rho)

    end_time = time.time()
    return selected_indices, end_time - start_time

def run_gasen_algo(probas, targets, nb_pru, args):
    """Wrapper for GASEN."""

    def vectorized_fitness(weights, proba, target):
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        norm_weights = weights / row_sums
        # (P, M) x (M, N, C) = (P, N, C)
        weighted_ensemble_proba = np.tensordot(norm_weights, proba, axes=([1],[0]))
        predictions = np.argmax(weighted_ensemble_proba, axis=2)
        errors = (predictions != target).mean(axis=1)
        return errors

    M, N, C = probas.shape
    
    # GA Config
    n_pop = 10
    n_generations = 20
    n_parents = 5
    
    start_time = time.time()
    
    # Init Population
    pop_shape = (n_pop, M)
    new_population = np.random.uniform(low=0, high=1, size=pop_shape)
    
    for gen in range(n_generations):
        fitness = GA.cal_pop_fitness(
            lambda pop: vectorized_fitness(pop, probas, targets), 
            new_population
        )
        parents = GA.select_mating_pool(new_population, fitness, n_parents)
        offspring_size = (n_pop - n_parents, M)
        offspring_cross = GA.crossover(parents, offspring_size)
        offspring_mut = GA.mutation(offspring_cross)
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mut

    final_fitness = GA.cal_pop_fitness(
        lambda pop: vectorized_fitness(pop, probas, targets), 
        new_population
    )
    
    best_idx = np.argmin(final_fitness)
    best_weights = new_population[best_idx]
    
    # Select top k
    selected_indices = np.argsort(best_weights)[::-1][:nb_pru]
    
    end_time = time.time()
    
    return selected_indices.tolist(), end_time - start_time

def run_pypruning_algo(algo_name, probas, targets, nb_pru, args):
    """Wrapper for PyPruning."""

    M, N, C = probas.shape
    
    start_time = time.time()
    
    pruner = create_pruner(algo_name, n_estimators=nb_pru)
    
    if pruner is None:
        raise ValueError(f"PyPruning algorithm '{algo_name}' returned None from create_pruner.")

    selected_indices, _ = pruner.prune_(probas, targets)
    
    end_time = time.time()
    
    # Convert numpy array to list if needed
    if isinstance(selected_indices, np.ndarray):
        selected_indices = selected_indices.tolist()
        
    return selected_indices, end_time - start_time

# --- MAIN LOGIC ---
def define_params():
    parser = argparse.ArgumentParser(description="Run Ensemble Pruning Ensemble Benchmark")
    
    # Data arguments
    parser.add_argument('--results', type=str, required=True, help='Path to tensor with inference results (M, N, C)')
    parser.add_argument('--targets', type=str, required=True, help='Path to tensor with targets (N,)')
    parser.add_argument('--outdir', type=str, default='.', help='Directory for output file(s)')

    # Pruning parameters
    parser.add_argument("--nb_pru", type=int, default=[5], nargs="+", help='Target size(s) of the pruned ensemble(s)')
    
    # Selection arguments
    parser.add_argument("--library", type=str, default='all', choices=['all', 'EPFD', 'GASEN', 'PyPruning'],
                        help='Select a specific library to run. Defaults to "all".')
    
    # Algorithm choices populated explicitly
    parser.add_argument("--algorithm", type=str, default=None, choices=ALL_ALGOS,
                        help='Run a single specific algorithm. Overrides --library.')

    # EPFD Specifics
    parser.add_argument("--distributed", action="store_true", help='Enable distributed mode for EPFD algorithms')
    parser.add_argument("--lam", type=float, default=0.5, help="Lambda parameter for EPFD")
    parser.add_argument("--m", type=int, default=2, help='Number of Machines (EPFD)')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')

    return parser.parse_args()

def main():
    args = define_params()

    # Load Data
    try:
        probas, targets, M, N, C = load_data(args.results, args.targets)
    except Exception as e:
        sys.exit(f"Error loading data: {e}")

    # Determine execution list
    tasks = [] # List of tuples (Library, Algorithm)

    # If specific algorithm is requested, find it
    if args.algorithm:
        found = False
        if args.algorithm in EPFD_ALGOS:
            tasks.append(("EPFD", args.algorithm))
            found = True
        elif args.algorithm in GASEN_ALGOS:
            tasks.append(("GASEN", args.algorithm))
            found = True
        elif args.algorithm in PYPRUNING_ALGOS:
            tasks.append(("PyPruning", args.algorithm))
            found = True
        
        if not found:
            sys.exit(f"Error: Algorithm '{args.algorithm}' not found in any library definitions.")
    else:
        # Otherwise append all algorithms from selected libraries
        if args.library in ['all', 'EPFD']:
            for alg in EPFD_ALGOS:
                tasks.append(("EPFD", alg))
        
        if args.library in ['all', 'GASEN']:
            for alg in GASEN_ALGOS:
                tasks.append(("GASEN", alg))
                
        if args.library in ['all', 'PyPruning']:
            for alg in PYPRUNING_ALGOS:
                tasks.append(("PyPruning", alg))

    # Target size can't be larger than ensemble size, no duplicate runs
    target_sizes = sorted(list(set([max(min(x,M),1) for x in args.nb_pru])))

    # Save start time for output creation
    slurm_job_id = os.getenv('SLURM_JOB_ID', default='')
    start_time = ('' if slurm_job_id != '' else time.strftime("%d-%m-%Y_%H-%M-%S"))
    
    for nb_pru in target_sizes:
        
        # Print run info
        print("\n" + "-" * 80)
        print(f"\n--- Starting Benchmark ---")
        print(f"Total Algorithms to run: {len(tasks)}")
        print(f"Target Ensemble Size: {nb_pru}")
        print(f"EPFD Distributed Mode: {args.distributed}")

        # Set Seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        pl.seed_everything(seed=args.seed)
        print("")

        results_data = []

        for lib, algo in tasks:
            print(f"Running [{lib}] {algo}...", end=" ", flush=True)
            
            try:
                runtime = 0.0
                selection = []
                
                if lib == "EPFD":
                    selection, runtime = run_epfd_algo(algo, probas, targets, nb_pru, args)
                elif lib == "GASEN":
                    selection, runtime = run_gasen_algo(probas, targets, nb_pru, args)
                elif lib == "PyPruning":
                    selection, runtime = run_pypruning_algo(algo, probas, targets, nb_pru, args)
                
                print(f"Done ({runtime:.4f}s)")
                results_data.append({
                    "Library": lib,
                    "Algorithm": algo,
                    "Runtime": runtime,
                    "Result": str(selection)
                })
                
            except Exception as e:
                print(f"FAILED. Error: {e}")
                results_data.append({
                    "Library": lib,
                    "Algorithm": algo,
                    "Runtime": -1.0,
                    "Result": f"Error: {str(e)}"
                })

        # Save Results
        # Save time for output creation
        slurm_job_id = os.getenv('SLURM_JOB_ID', default='')
        slurm_array_job_id = os.getenv('SLURM_ARRAY_JOB_ID', default='')
        end_time = ('' if slurm_job_id != '' else time.strftime("%d-%m-%Y_%H-%M-%S"))

        try:
            if len(results_data) == 1:
                outdir = Path(args.outdir).resolve() / "pruning" / f"{slurm_array_job_id}" / f"{nb_pru}" / f"{results_data[0]['Library']}" / f"{results_data[0]['Algorithm']}" / f"{slurm_job_id}" / f"{end_time}" 
            else:
                outdir = Path(args.outdir).resolve() / "pruning" / f"{slurm_array_job_id}" / f"{nb_pru}" / f"{slurm_job_id}" / f"{end_time}" 

            outdir.mkdir(exist_ok=True, parents=True)
            # Save to CSV
            df = pd.DataFrame(results_data)
            df.to_csv(outdir / f'results.csv', index=False)
            
            # Save to HTML
            html_string = df.to_html(classes='table table-striped', index=False)
            final_html = f"""
            <html>
            <head>
                <title>Ensemble Pruning Benchmark Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h2>Ensemble Pruning Results</h2>
                <p><strong>Input:</strong> {args.results}</p>
                <p><strong>Input Shape:</strong> (M={M}, N={N}, C={C})</p>
                <p><strong>Target Size:</strong> {nb_pru}</p>
                <br>
                {html_string}
            </body>
            </html>
            """
            
            with open(outdir / f'results.html', 'w') as f:
                f.write(final_html)
                
            print(f"\nResults successfully saved to {outdir}")
            
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
