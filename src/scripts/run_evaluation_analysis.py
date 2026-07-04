# === IMPORTS ===
# stdlib imports
import logging
import argparse
import sys
import time
from copy import deepcopy

# 3-rd party imports
from pathlib import Path
from alive_progress import alive_bar, alive_it
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import numpy as np

# local imports
from shared_utils.custom_logging import create_logger


# === GLOBAL VARIABLES ===
logger = create_logger("Analyse Evaluation Results Logger", log_level=logging.INFO)
start_time = time.strftime('%d-%m-%Y_%H-%M-%S')

EPFD_ALGOS = [
        "ES", "KP", "KL", "RE", "OO", "SEP", "OEP", "PEP", 
        "GMM", "LCS", "COMEP", "DOMEP",
]

GASEN_ALGOS = ["GASEN"]

PYPRUNING_ALGOS = [
        "individual_margin_diversity", "individual_contribution", "individual_error",
        "individual_kappa_statistic", "reduced_error", "complementariness", "drep",
        "margin_distance", "combined", "reference_vector", "combined_error",
        "error_ambiguity", "largest_mean_distance", "cluster_accuracy", 
        "cluster_centroids"
]

BASELINE_ALGOS = [
        "bsm",
        "wsm",
        "random",
        "random_no_replacement",
]

ALL_ALGOS = EPFD_ALGOS + GASEN_ALGOS + PYPRUNING_ALGOS + BASELINE_ALGOS

ALL_TYPES = [
        "all", 
        "greedy",
        "ranking", 
        "optimization", 
        "clustering",
]

TYPE_OF = {
        # Library categorizes | Library implements | Algorithm is
        "ES": "ranking",  # ranking | ranking | ranking
        "KP": "ranking", # ranking | ranking | ranking
        "KL": "greedy", # ranking | greedy | greedy 
        "RE": "greedy", # ranking | greedy | greedy
        "OO": "ranking", # ranking | ranking | ranking 
        "SEP": "optimization", # optimization | optimization | optimization
        "OEP": "greedy", # ranking | greedy | greedy
        "PEP": "optimization", # optimization | optimization | optimization
        "GMM": "clustering",  # composable | clustering | clustering / greedy
        "LCS": "greedy", # composable | greedy | greedy
        "COMEP": "greedy", # entropy | greedy | greedy
        "DOMEP": "greedy", # entropy | greedy | greedy
        "GASEN": "optimization", # optimization | optimization | optimization
        "individual_margin_diversity": "ranking",  # ranking | ranking | ranking
        "individual_contribution": "ranking", # ranking | ranking | ranking
        "individual_error": "ranking", # ranking | ranking | ranking
        "individual_kappa_statistic": "ranking", # ranking | ranking | ranking / greedy
        "reduced_error": "greedy", # greedy | greedy | greedy
        "complementariness": "greedy", # greedy | greedy | greedy
        "drep": "greedy", # greedy | greedy | greedy
        "margin_distance": "greedy", # greedy | greedy | greedy
        "combined": "optimization", # optimization | optimization | optimization
        "reference_vector": "ranking", # ranking | ranking | ranking
        "combined_error": "optimization", # optimization | optimization | optimization
        "error_ambiguity": "ranking", # ranking | ranking | greedy
        "largest_mean_distance": "clustering", # clustering | clustering | clustering
        "cluster_accuracy": "clustering", # clustering | clustering | clustering 
        "cluster_centroids": "clustering", # clustering | clustering | clustering
        "random": "baseline", # random | random | baseline
        "bsm": "baseline", # baseline | ranking | ranking
        "wsm": "baseline", # baseline | ranking | ranking
        "random_no_replacement": "baseline", # baseline | random | baseline
}

GLOBAL_METRICS = {
    "Cohen`s Kappa": {
        "k": "",
        "Expected Accuracy": "",
        "Observed Accuracy": "",
        "Accuracy Gain<br>(over \"Expected Accuracy\")": "",
    },
    "Matthews Correlation Coefficient": {
        "": "",
    },
    "Average One-vs-Rest ROC AUC": {
        "Micro": "",
        "Macro": "",
        "Weighted": "",
    },
    "Average One-vs-Rest PR AUC<br><br>(Average Precision Score)": {
        "Micro": "",
        "Macro": "",
        "Weighted": "",
    },
    "Brier Score Loss": {
        "Micro (Overall)": "",
        "Macro": "",
        "Weighted": "",
    },
    "Expected Calibration Error": {
        "": "",
    },
    "Average Accuracy": {
        "Micro<br>(= Micro Precision, Recall, F1-Score)": "",
        "Macro<br>(= Macro Recall)": "",
        "Weighted<br>(= Weighted Recall)": "",
    },
    "Average Precision": {
        "Macro": "",
        "Weighted": "",
    },
    "Average Recall": {
        "Macro": "",
        "Weighted": "",
    },
    "Average F1-Score": {
        "Macro": "",
        "Weighted": "",
    },
}

# === HELPER FUNCTIONS ===
def parse_args():

    # init parser
    parser = argparse.ArgumentParser()

    # add args
    parser.add_argument("evaluation_dir", help="Path to directory with evaluation results")
    parser.add_argument("out_dir", help="Path to out-directory (files will be saved under: 'out_dir/analyses/d-m-Y_H-M-S/metric')")
    parser.add_argument("--debug", help="Print additional debug info", action="store_true")

    # parse args
    args = parser.parse_args()

    # change log level if debug flag was given
    if args.debug:
        global logger
        logger.info("Log-Level set to DEBUG...")
        logger = create_logger("Analyse Evaluation Results Logger (DEBUG)", log_level=logging.DEBUG)
    else:
        logger.info("Log-Level set to INFO...")


    # print info about parsed args
    logger.info(f"Parsed the following arguments:")
    for key, val in vars(args).items():
        logger.info(f"{key}: {val}")

    # return namespace object
    return args

def sanitize(string: str, save_conform: bool = False):
    if save_conform:
        return string.replace(", ","-").replace(" ", "_").replace("`", "").replace("<br>", "").replace('"', '').replace("(", "_").replace(")", "").replace("=","eq").lower()
    else: 
        return string.replace("<br>", " ", 1).replace("<br>", "")


# === MAIN EXECUTION LOOP ===
def main():

    # parse args
    args = parse_args()
    
    # resolve dirs to path
    try:
        # get dirs and resolve
        args.evaluation_dir = Path(args.evaluation_dir).resolve()
        args.out_dir = Path(args.out_dir).resolve()
        dirs = [args.evaluation_dir, args.out_dir]

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
        evaluation_results = [x for x in args.evaluation_dir.rglob(f"metrics_results.json")]
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
            if result.is_relative_to(f"{result.parents[1]}/{algo}"):
                # get the subensemble_size
                subensemble_size = int(result.parents[2].stem)
                # save size and metric under the algos list
                algo_bins[algo].append({"Size": subensemble_size, "Metrics": pd.read_json(result)})
    
    logger.info(f"Sorting results by ascending ensemble-size within each bin...")
    # sort each algo bins' contents by size
    for algo, _ in alive_it(algo_bins.items()):
        algo_bins[algo] = sorted(algo_bins[algo], key=lambda x: x["Size"])

    # assign each algo bin to a type
    types = {type: deepcopy(BASELINE_ALGOS) for type in ALL_TYPES} # every type of pruning algorithm has all baseline algos
    for algo, _ in algo_bins.items():
        if algo not in BASELINE_ALGOS: # prevent baseline from redundantly being added here
            types['all'].append(algo) # the all type get's all algos 
            types[TYPE_OF[algo]].append(algo) # the specific algo gets assigned to its type

    # plot results
    for type, algos_of_type in types.items():
        logger.info(f"[{type}] - Creating combined plots...")

        total_algos_of_type = len(algos_of_type)

        # get distinct colours
        colours = plt.get_cmap("hsv", total_algos_of_type)

        # plot each metric over ensemble size
        for metric, submetrics in alive_it(GLOBAL_METRICS.items()):
            for submetric, _ in submetrics.items():

                # convert strings
                metric_san = sanitize(metric)
                submetric_san = f"({sanitize(submetric)})" if submetric else ""
                logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - Creating plot...")

                # prepare figure
                plt.figure(figsize=(12,16))
                empty_figure = True

                # set custom x ticks
                custom_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 75, 100]

                # always plot full ensemble performance as baseline
                full_ensemble_val = None
                if "random" in algo_bins:
                    for results in algo_bins["random"]:
                        if int(results["Size"]) == 100:
                            full_ensemble_val = float(results["Metrics"][metric][submetric])
                            break
                if full_ensemble_val is not None:
                    plt.plot(range(len(custom_ticks)), [full_ensemble_val] * len(custom_ticks), label="full ensemble", color="grey", linestyle=":")

                # track overlapping markers
                overlap_tracker = {}
                text_tracker = {}
                segment_tracker = []

                # add each metric for all algos to the plot
                for idx, algo in enumerate(algos_of_type):
                    results = [x for x in algo_bins[algo] if int(x["Size"]) <= 75]
                    if results:
                        logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Creating curve...")
                        # place all sizes and metric values for selected algo in lists
                        subensemble_sizes = [int(x["Size"]) for x in results]
                        subensemble_metrics = [float(x["Metrics"][metric][submetric]) for x in results]
                        if algo in ['bsm','wsm']:
                            # only consider size one result for bsm and wsm
                            subensemble_sizes = range(len(custom_ticks))
                            subensemble_metrics = [subensemble_metrics[0]] * len(custom_ticks)
                            logger.debug(f"subensemble_sizes: {subensemble_sizes}")
                            logger.debug(f"subensemble_metrics: {subensemble_metrics}")
                            if algo == 'bsm':
                                plt.plot(subensemble_sizes, subensemble_metrics, label=f"{algo}", color="grey", linestyle="--")
                            else: 
                                plt.plot(subensemble_sizes, subensemble_metrics, label=f"{algo}", color="grey", linestyle="-.")
                        else:
                            subensemble_sizes = np.interp(subensemble_sizes, custom_ticks, range(len(custom_ticks))) 
                            plt.scatter(subensemble_sizes, subensemble_metrics, color=colours(idx))
                            
                            # handle overlapping lines
                            linestyle="-"
                            coords = list(zip(subensemble_sizes, np.round(subensemble_metrics, 3)))
                            current_segments = set(zip(coords[:-1], coords[1:]))
                            for saved_segments in segment_tracker:
                                if len(current_segments & saved_segments) >= 1:
                                    linestyle = (0,(5,5))
                                    break
                            segment_tracker.append(current_segments)
                            plt.plot(subensemble_sizes, subensemble_metrics, label=f"{algo}", color=colours(idx), linestyle=linestyle, zorder=4)

                            # handle peaks
                            if "Loss" in metric or "Error" in metric:
                                best_idx = np.argmin(subensemble_metrics)
                            else:
                                best_idx = np.argmax(subensemble_metrics)

                            peak_coord = (subensemble_sizes[best_idx], round(subensemble_metrics[best_idx], 3))

                            # handle overlaps
                            overlaps = overlap_tracker.get(peak_coord, 0)
                            #fillstyle = None
                            #markersize = 15
                            #if overlaps == 0:
                            #    zorder = 5
                            #    fillstyle = 'full'
                            #    markersize = 20
                            #elif overlaps == 1:
                            #    zorder = 6
                            #    fillstyle = 'left'
                            #    logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Handling first peak overlap at {algo} in {type}...")
                            #elif overlaps == 2:
                            #    zorder = 7
                            #    fillstyle = 'right'
                            #    logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Handling second peak overlap at {algo} in {type}...")
                            #elif overlaps == 3:
                            #    zorder = 8
                            #    fillstyle = 'bottom'
                            #    logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Handling third peak overlap at {algo} in {type}...")
                            #else:
                            #    zorder = 9
                            #    logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Handling {overlaps}th peak overlap at {algo} in {type} (updating text)...") 
                            fillstyle = 'full'
                            if overlaps == 0:
                                marker = 7
                            elif overlaps == 1:
                                marker = 4
                                logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Handling first peak overlap at {algo} in {type}...")
                            elif overlaps == 2:
                                marker = 6
                                logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Handling second peak overlap at {algo} in {type}...")
                            elif overlaps == 3:
                                marker = 5
                                logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Handling third peak overlap at {algo} in {type}...")
                            else:
                                fillstyle = None
                                logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - [{idx+1}/{total_algos_of_type}] - [{algo}] - Handling {overlaps}th peak overlap at {algo} in {type} (updating text)...") 

                            #if overlaps == 0:
                            #    t = None
                            #else:
                            #    t = Affine2D().rotate_deg(45/overlaps)
                            #plt.plot(peak_coord[0], peak_coord[1], 
                            #         marker=MarkerStyle('x', transform=t),
                            #         markersize=15, 
                            #         linestyle='None', 
                            #         color=colours(idx), 
                            #         #markeredgecolor='Black', 
                            #         #markeredgewidth=0,
                            #         #fillstyle=fillstyle, 
                            #         zorder=5)
                            
                            if fillstyle:
                                plt.plot(subensemble_sizes[best_idx], subensemble_metrics[best_idx], 
                                         marker=marker,
                                         markersize=15, 
                                         linestyle='None', 
                                         color=colours(idx), 
                                         markeredgecolor=None,
                                         markeredgewidth=0,
                                         fillstyle=fillstyle, 
                                         zorder=5)
                            else: 
                                total_algos = overlaps + 1
                                # if text exists, update it
                                if peak_coord in text_tracker:
                                    text_tracker[peak_coord].set_text(str(total_algos))
                                # otherwise create it
                                else:
                                    txt = plt.text(subensemble_sizes[best_idx], subensemble_metrics[best_idx], 
                                                   str(total_algos), ha='center', va='center', 
                                                   fontsize=10, color='white', fontweight='bold', zorder=6)
                                    text_tracker[peak_coord] = txt


                            overlap_tracker[peak_coord] = overlaps + 1

                        empty_figure = False

                if not empty_figure:
                    # finalize plot
                    plt.xticks(ticks=range(len(custom_ticks)), labels=[str(t) for t in custom_ticks])
                    plt.xlim(0,len(custom_ticks)-1)

                    if submetric_san:
                        plt.ylabel(f"{metric_san} {submetric_san}", fontsize="x-large")
                    else: 
                        plt.ylabel(f"{metric_san}", fontsize="x-large")
                    plt.xlabel(f"Ensemble-Size", fontsize="x-large")
                    plt.plot([], [], marker=7, color='white', markeredgecolor='black', 
                             markersize=10, linestyle='None', label='Peak Performance')
                    plt.legend(loc="best", fontsize="medium", ncol=1)
                    plt.title(f"{type.title()} Pruning Techniques - {metric_san} {submetric_san} vs. Ensemble-Size", fontsize="x-large")
                    # save
                    logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - Saving plot...")
                    try: 
                        # sanitize strings
                        metric_save = sanitize(metric, save_conform=True)
                        submetric_save = sanitize(submetric, save_conform=True)
                        # create filename
                        filename = f"{submetric_save}.png" if submetric_save else f"out.png"
                        # create outdir
                        out_dir = args.out_dir / "analyses" / f"{start_time}" / f"{type}" / f"{metric_save}"
                        out_dir.mkdir(exist_ok=True, parents=True)
                        # save figure
                        plt.savefig(out_dir / filename)
                        plt.close()
                        logger.info(f"[{type}] - [{metric_san}] - [{submetric_san}] - Plot saved under: {out_dir}/{filename}...")
                    except Exception as e:
                        logger.critical(f"Error while saving plot for {metric_san} {submetric_san}: {e}")
                        raise
                        sys.exit(1)
                else: 
                    logger.warning("Had nothing to plot")
    logger.info("FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
