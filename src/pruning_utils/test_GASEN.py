# system
import sys

# 3rd party
from pathlib import Path
import numpy as np
import torch

# local 
import gasen.GA as GA

def vectorized_fitness(weights, proba, target):
    """
    Calculates ensemble error rate (1 - accuracy) using Soft Voting.
    
    Args:
        weights: (P, M) Population weights. P=Population-Size, M=Models.
        proba:   (M, N, C) Inference probabilities. N=Samples, C=Classes.
        target:  (N,) True labels.
    Returns:
        (P,) Error rate per individual.
    """
    # Normalize weights (L1) -> Shape (P, M)
    # Sum over models (axis 1) and divide to treat as voting power
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 # avoid 0 divison
    norm_weights = weights / row_sums

    # Weighted Soft Voting -> Shape (P, N, C)
    # Tensor Dot: (P, M) x (M, N, C) = (P, N, C)
    # Combines voting power (weights) with votes (proba) along Model axis
    weighted_ensemble_proba = np.tensordot(norm_weights, proba, axes=([1],[0]))

    # Accuracy Calculation
    # Argmax over classes (axis 2) -> (P, N) predictions
    predictions = np.argmax(weighted_ensemble_proba, axis=2)
    
    # Compare to target (broadcasted) -> (P, N) booleans
    # Mean over samples (axis 1) -> (P,) scalar error
    errors = (predictions != target).mean(axis=1)

    return errors

def main():
    try:
        # Load tensors
        proba_path = Path(sys.argv[1]).resolve()
        target_path = Path(sys.argv[2]).resolve()
        
        proba = torch.load(proba_path).float().numpy()
        target = torch.load(target_path).float().numpy()
    except Exception as e:
        print(f"Usage: python GASEN_test.py <proba.pt> <target.pt>\nError: {e}")
        sys.exit(1)

    # Dimensions: M=Models, N=Samples, C=Classes
    M, N, C = proba.shape
    
    # GA Config
    n_keep = 2          # Top K models to select
    n_pop = 10          # Population size
    n_generations = 20
    n_parents = 5
    
    # Init Population (P, M)
    pop_shape = (n_pop, M)
    new_population = np.random.uniform(low=0, high=1, size=pop_shape)
    
    print(f"\nStarting GASEN (M={M}, N={N}, #GEN={n_generations})...")
    
    for gen in range(n_generations):
        # Fitness: Pass lambda to inject data
        fitness = GA.cal_pop_fitness(
            lambda pop: vectorized_fitness(pop, proba, target), 
            new_population
        )
        
        # Standard GA steps
        parents = GA.select_mating_pool(new_population, fitness, n_parents)
        
        offspring_size = (n_pop - n_parents, M)
        offspring_cross = GA.crossover(parents, offspring_size)
        offspring_mut = GA.mutation(offspring_cross)
        
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mut

    # Final Evaluation
    final_fitness = GA.cal_pop_fitness(
        lambda pop: vectorized_fitness(pop, proba, target), 
        new_population
    )
    
    best_idx = np.argmin(final_fitness)
    best_weights = new_population[best_idx]
    
    # Pruning: Select indices of highest weights
    selected_indices = np.argsort(best_weights)[::-1][:n_keep]
    
    print(f"\nBest Error Rate: {final_fitness[best_idx]:.4f}")
    print(f"Selected Indices: {selected_indices.tolist()}")
    print(f"Selected Weights: {best_weights[selected_indices]}")

if __name__ == "__main__":
    main()
