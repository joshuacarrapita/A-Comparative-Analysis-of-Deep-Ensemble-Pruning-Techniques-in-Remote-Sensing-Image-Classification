import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# --- LIBRARY ALGORITHM DEFINITIONS ---
# Defined globally to be used in argparse choices
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

# --- LIBRARY IMPORTS ---

# 1. EPFD Imports
try:
    from epfd.data_distributed import (distributed_single_pruning,
                                       distributed_pruning_methods,
                                       COMEP_Pruning, DOMEP_Pruning)
    EPFD_AVAILABLE = True
except ImportError:
    EPFD_AVAILABLE = False
    print("Warning: EPFD library not found.")

# 2. GASEN Imports
try:
    import gasen.GA as GA
    GASEN_AVAILABLE = True
except ImportError:
    GASEN_AVAILABLE = False
    print("Warning: GASEN library not found.")

# 3. PyPruning Imports
try:
    # Trying both import structures depending on how the module is installed/located
    try:
        from PyPruning.Papers import create_pruner
    except ImportError:
        from PyPruning.PyPruning.Papers import create_pruner
    PYPRUNING_AVAILABLE = True
except ImportError:
    PYPRUNING_AVAILABLE = False
    print("Warning: PyPruning library not found.")
    # Define dummy to prevent NameError if logic flows there incorrectly
    def create_pruner(*args, **kwargs): 
        raise ImportError("PyPruning library is not installed or accessible.")

# --- HELPER FUNCTIONS ---

def load_data(results_path, targets_path):
    """Loads tensors and infers dimensions M, N, C."""
    print(f"Loading data from:\n  Probas: {results_path}\n  Targets: {targets_path}")
    
    probas = torch.load(Path(results_path).resolve()).float().numpy()
    targets = torch.load(Path(targets_path).resolve()).float().numpy()
    
    # User confirmed input is always (M, N, C)
    if len(probas.shape) == 3:
        M, N, C = probas.shape
    else:
        # Fallback if dimensions look swapped, but prioritizing user instruction
        print(f"Warning: Unexpected shape {probas.shape}. Assuming (M, N, C) as requested.")
        M, N, C = probas.shape

    print(f"Inferred Dimensions -> Models (M): {M}, Samples (N): {N}, Classes (C): {C}")
    return probas, targets, M, N, C

# --- ALGORITHM IMPLEMENTATIONS ---

def run_epfd_algo(algo_name, probas, targets, nb_pru, args):
    """Wrapper for EPFD library logic."""
    if not EPFD_AVAILABLE:
        raise ImportError("EPFD Library not loaded")

    M, N, C = probas.shape
    
    # 1. Get Hard Labels
    # Input is (M, N, C). argmax over classes (axis 2) -> (M, N)
    results_hard = np.argmax(probas, axis=2) 
    
    rho = nb_pru / M
    selected_indices = []

    start_time = time.time()
    
    if algo_name in ['COMEP', 'DOMEP']:
        # Original test_EPFD.py used: COMEP_Pruning(np.array(results).T.tolist(), ...)
        # Since we have results_hard as (M, N), .T becomes (N, M). 
        # This matches the explicit transposition in the provided test file.
        results_transposed_list = results_hard.T.tolist()

        if algo_name == 'COMEP':
            selected_indices = COMEP_Pruning(results_transposed_list, nb_pru, targets, args.lam)
        elif algo_name == 'DOMEP':
            selected_indices = DOMEP_Pruning(results_transposed_list, nb_pru, args.m, targets, args.lam)
            
    else:
        # Single pruning methods (e.g., 'ES', 'KP'...)
        # Original test_EPFD.py used: distributed_single_pruning(name, targets, results, ...)
        # It did NOT use .T here. We pass results_hard (M, N) directly.
        
        if args.distributed:
            # Distributed wrapper in test_EPFD usually implies the framework logic
            # The test file calls `distributed_pruning_methods` for distributed
            # Note: variable names in test_EPFD were `results` passed directly.
            selected_indices = distributed_pruning_methods(targets, results_hard, nb_pru, args.m, algo_name, rho=rho)
        else:
            selected_indices = distributed_single_pruning(algo_name, targets, results_hard, M, nb_pru, rho=rho)

    end_time = time.time()
    return selected_indices, end_time - start_time

def run_gasen_algo(probas, targets, nb_pru, args):
    """Wrapper for GASEN."""
    if not GASEN_AVAILABLE:
        raise ImportError("GASEN Library not loaded")

    # Internal helper from test_GASEN.py
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
    if not PYPRUNING_AVAILABLE:
        raise ImportError("PyPruning Library not loaded")

    M, N, C = probas.shape
    
    start_time = time.time()
    
    # Create pruner
    # Using the imported create_pruner function
    pruner = create_pruner(algo_name, n_estimators=nb_pru)
    
    if pruner is None:
        raise ValueError(f"PyPruning algorithm '{algo_name}' returned None from create_pruner.")

    # Execute
    # PyPruning inputs: The library typically handles shapes robustly, 
    # but based on test_PyPruning.py, it takes probas directly.
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
    parser.add_argument('--output', type=str, default='pruning_results.html', help='Path for output HTML file')

    # Pruning parameters
    parser.add_argument("--nb_pru", type=int, default=5, help='Target size of the pruned ensemble')
    
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
    
    # Set Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        
        # Note: Since we used `choices` in argparse, the script would likely 
        # exit before this if an invalid name is passed, but good for safety.
        if not found:
            sys.exit(f"Error: Algorithm '{args.algorithm}' not found in any library definitions.")
    else:
        # Based on Library selection
        if args.library in ['all', 'EPFD']:
            for alg in EPFD_ALGOS:
                tasks.append(("EPFD", alg))
        
        if args.library in ['all', 'GASEN']:
            for alg in GASEN_ALGOS:
                tasks.append(("GASEN", alg))
                
        if args.library in ['all', 'PyPruning']:
            for alg in PYPRUNING_ALGOS:
                tasks.append(("PyPruning", alg))

    print(f"\n--- Starting Benchmark ---")
    print(f"Total Algorithms to run: {len(tasks)}")
    print(f"Target Ensemble Size: {args.nb_pru}")
    print(f"EPFD Distributed Mode: {args.distributed}")
    print("-" * 60)

    results_data = []

    for lib, algo in tasks:
        print(f"Running [{lib}] {algo}...", end=" ", flush=True)
        
        try:
            runtime = 0.0
            selection = []
            
            if lib == "EPFD":
                selection, runtime = run_epfd_algo(algo, probas, targets, args.nb_pru, args)
            elif lib == "GASEN":
                selection, runtime = run_gasen_algo(probas, targets, args.nb_pru, args)
            elif lib == "PyPruning":
                selection, runtime = run_pypruning_algo(algo, probas, targets, args.nb_pru, args)
            
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
    df = pd.DataFrame(results_data)
    
    # Save to HTML
    try:
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
            <p><strong>Target Size:</strong> {args.nb_pru}</p>
            <br>
            {html_string}
        </body>
        </html>
        """
        
        with open(args.output, 'w') as f:
            f.write(final_html)
            
        print(f"\nResults successfully saved to {args.output}")
        
    except Exception as e:
        print(f"Error saving HTML: {e}")

if __name__ == "__main__":
    main()
