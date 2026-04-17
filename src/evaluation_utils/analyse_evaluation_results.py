# === IMPORTS ===
# stdlib imports
import argparse
import sys

# 3-rd party imports
from pathlib import Path
from alive_progress import alive_bar, alive_it
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# local imports
from shared_utils.custom_logging import create_logger


# === GLOBAL VARIABLES ===
logger = create_logger("Analyse Evaluation Results Logger")

EPFD_ALGOS = [
    "ES", "KP", "KL", "RE", "OO", "DREP", "SEP", "OEP", "PEP", 
    "GMA", "LCS", "COMEP", "DOMEP"
]
GASEN_ALGOS = ["GASEN"]
PYPRUNING_ALGOS = [
    "individual_margin_diversity", "individual_contribution", "individual_error",
    "individual_kappa_statistic", "reduced_error", "complementariness", "drep",
    "margin_distance", "combined", "reference_vector", "combined_error",
    "error_ambiguity", "largest_mean_distance", "cluster_accuracy", 
    "cluster_centroids", "random"
]

ALL_ALGOS = EPFD_ALGOS + GASEN_ALGOS + PYPRUNING_ALGOS

GLOBAL_METRICS = [ 
        "Cohen`s Kappa",
        "Matthews Correlation Coefficient",
        "Micro-Average One-vs-Rest ROC AUC",
        "Macro-Average One-vs-Rest ROC AUC",
        "Weighted-Average One-vs-Rest ROC AUC",
        "Micro-Average One-vs-Rest PR AUC",
        "Macro-Average One-vs-Rest PR AUC",
        "Weighted-Average One-vs-Rest PR AUC",
        "Brier Score Loss",
        "Expected Calibration Error",
]

# === HELPER FUNCTIONS ===
def parse_args():

    # init parser
    parser = argparse.ArgumentParser()

    # add args
    parser.add_argument("evaluation_dir", help="Path to directory with evaluation results")
    parser.add_argument("out_dir", help="Path to out-directory (files will be saved under: 'out_dir/analyses/d-m-Y_H-M-S/metric')")

    # parse args
    args = parser.parse_args()

    # print info about parsed args
    logger.info(f"Parsed the following arguments:")
    for key, val in vars(args).items():
        logger.info(f"{key}: {val}")

    # return namespace object
    return args


# === MAIN EXECUTION LOOP ===
def main():

    # parse args
    args = parse_args()
    
    # TODO: create plots for each pruning algorithm and each global value
    #       over all ensemble sizes

    # resolve dirs to path
    try:
        # get dirs and resolve
        evaluation_dir = Path(args.evaluation_dir).resolve()
        out_dir = Path(args.out_dir).resolve()
        dirs = [evaluation_dir, out_dir]

        # check dirs formats
        for dir in dirs:
            if not dir.is_dir():
                if not dir.exists():
                    raise FileNotFoundError(f"Directory not found: {dir}")
                else:
                    raise NotImplementedError(f"Not a directory: {dir}")
    except (FileNotFoundError, NotImplementedError) as e:
        logger.critical(f"{e}")
        raise
        sys.exit(1)

    # retrieve evaluation results
    with alive_bar(1) as bar:
        logger.info(f"Retrieving evaluation results...")
        evaluation_results = [x for x in evaluation_dir.rglob(f"metrics_results.json")]
        bar()

    # filter evaluation_results into algo bins
    logger.info(f"Filtering evaluation results into bins of algorithms:\n{ALL_ALGOS}...")
    # create bin for each algo
    algo_bins = {algo: [] for algo in ALL_ALGOS}
    # for each metrics_results.json
    for result in alive_it(evaluation_results):    
        # for each algo
        for algo in ALL_ALGOS:
            # if metrics_results.json belongs to this algo
            if result.is_relative_to(f"{result.parents[3]}/{algo}"):
                # get the subensemble_size
                subensemble_size = int(result.parents[4].stem)
                # save both under the algos list
                algo_bins[algo].append({"Size": subensemble_size, "Metrics": pd.read_json(result)})
    
    logger.info(f"Sorting results by ascending ensemble-size within each bin...")
    # sort each algo bins' contents by size
    for algo, _ in alive_it(algo_bins.items()):
        algo_bins[algo] = sorted(algo_bins[algo], key=lambda x: x["Size"])

    # plot results
    logger.info(f"Creating plots...")
    # create figure
    plt.figure(figsize=(12,16))
    empty_figure = True
    # get distinct colours
    colours = plt.get_cmap("hsv", len(GLOBAL_METRICS))
    # plot each metric over ensemble size
    total_algos = len(algo_bins)
    for idx, metric in enumerate(alive_it(GLOBAL_METRICS)):
        logger.info(f"Creating plot for {metric}...")
        # add each metric for all algos to the plot
        for jdx, (algo, results) in enumerate(algo_bins.items()):
            if results:
                logger.info(f"({jdx}/{total_algos}) Creating curve for {algo}...")
                # place all sizes and metric values for selected algo in lists
                subensemble_sizes = [int(x["Size"]) for x in results]
                print(results[0]["Metrics"])
                subensemble_metrics = [int(x["Metrics"][metric]['k']) for x in results]
                # interpolate function over [0,100]
                interp_grid = np.linspace(0,100,100)
                interp_metrics = np.interp(interp_grid, subensemble_size, subensemble_metrics)
                # values to plot
                plt.plot(interp_grid, interp_metrics, label=f"{algo}", color=colours(idx))
                empty_figure = False
        if not empty_figure:
            # finalize plot
            plt.legend(loc="best", fontsize="medium", ncol=1)
            plt.title(f"{metric} vs. Ensemble-Size", fontsize="x-large")
            # save
            logger.info(f"Saving plot for {metric}...")
            try: 
                filename = f"{metric.replace(' ', '_').lower()}.png"
                out_dir = out_dir / analyses / f"{time.strftime('%d-%m-%Y_%H-%M-%S')}" / f"{metric}"
                out_dir.mkdir(exist_ok=True, parents=True)
                plt.savefig(out_dir / filename)
            except Exception as e:
                logger.critical(f"Error while saving plot for metric {metric}: {e}")
                raise
                sys.exit(1)
        else: 
            logger.warning("Had nothing to plot")

if __name__ == "__main__":
    main()
