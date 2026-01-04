# TODO: Add imports
# std imports

# 3rd party imports
import lightning.pytorch as pl
from sklearn.metrics import confusion_matrix

# local imports
from shared_utils.custom_logging import create_logger

# --- GLOBAL DEFINITIONS ---

# --- create global logger ---
if __debug__:
    logger = create_logger("EnsembleEvaluationLogger", log_level=logging.DEBUG)
    logger.newline()
    logger.debug('--- DEBUG ON -- (Log level: DEBUG)')
    logger.newline()
    util.MESSAGE_LEVEL = util.MessageLevel.INFO
else:
    logger = create_logger("EnsembleEvaluationLogger", log_level=logging.INFO)
    logger.newline()
    logger.info('--- DEBUG OFF --- (Log level: INFO)')
    logger.newline()

# --- metrics ---
CLASS-WISE_METRICS = [
        'Precision', 
        'Recall', 
        'Specificity', 
        'F1-Score', 
]

GLOBAL_METRICS = [
        'Macro-Average', 
        'Micro-Average', 
        'Weighted-Average', 
        'Cohen`s Kappa', 
        'Matthews Correlation Coefficient', 
]

VISUALIZATIONS = [
        'Row-normalized Heatmap', 
        'Column-normalized Heatmap', 
        'One-vs-Rest ROC',
        'One-vs-One ROC', 
        'Macro-averaged ROC', 
        'Precision-Recall Curve',
]

ALL_METRICS = CLASS-WISE_METRICS + GLOBAL_METRICS + VISUALIZATIONS

# --- HELPER FUNCTIONS ---
def load_data(results_path, targets_path):
    """Loads tensors and infers dimensions M, N, C."""
    print(f"\nLoading data from:\n  Probas: {results_path}\n  Targets: {targets_path}")
    
    probas = torch.load(Path(results_path).resolve()).float().numpy()
    targets = torch.load(Path(targets_path).resolve()).float().numpy()
    
    M, N, C = probas.shape

    print(f"Inferred Dimensions -> Models (M): {M}, Samples (N): {N}, Classes (C): {C}")
    return probas, targets, M, N, C

def ensemble_decision(probas, targets):
    # TODO derive the ensemble_decision based on e.g. majority vote

def define_params():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble Inference Results')
    
    # Data arguments
    parser.add_argument('--results', type=str, required=True, help='Path to tensor with inference results (M, N, C)')
    parser.add_argument('--targets', type=str, required=True, help='Path to tensor with targets (N,)')
    parser.add_argument('--outdir', type=str, default='.', help='Directory for output file(s)')

    # Metric arguments
    parser.add_argument('--metric-types', type=str, default='all', choices=['all', 'class-wise', 'global', 'visualization'], nargs='+', help='Select metric type(s) to evaluate with. Defaults to "all".')
    parser.add_argument('--metrics', type=str, default='', choices=ALL_METRICS, nargs="+", help='Evaluate using specific metric(s). Overrides --metric-types.')
    
    # Seed
    parser.add_argument('--seed', type=int, default=666, help='Random seed')

    return parser.parse_args()

# --- MAIN LOGIC ---
def main():
    args = define_params()

    # Load Data
    try:
        probas, targets, M, N, C = load_data(args.results, args.targets)
    except Exception as e:
        logger.critical(f'Error loading data: {e}')
        sys.exit(1)

    # Determine execution list
    tasks = [] # List of tuples (Metric-Type, Metric)

    # If specific metrics are requested, find them
    if args.metrics:
        not_found = []
        for idx, metric in enumerate(args.metrics):
            if metric in CLASS-WISE_METRICS:
                tasks.append(("CLASS-WISE", args.metrics[idx]))
            elif metric in GLOBAL_METRICS:
                tasks.append(("GLOBAL", args.metrics[idx]))
            elif metric in VISUALIZATIONS:
                tasks.append(("VISUALIZATIONS", args.metrics[idx]))
            else:
                not_found.append(args.metrics[idx])
        if not_found:
            logger.newline()
            logger.warning(f'Warning: The following metrics could not be found: {not_found}. Continuing without these metrics.')
            logger.newline()
    else:
        # Otherwise append all algorithms from selected libraries
        if args.metric-types in ['all', 'class-wise']:
            for metric in CLASS-WISE_METRICS:
                tasks.append(('class-wise', metric))
        if args.library in ['all', 'global']:
            for metric in GLOBAL_METRICS:
                tasks.append(('global', metric))
        if args.library in ['all', 'visualization']:
            for metric in VISUALIZATIONS:
                tasks.append(('visualization', metric))

    results_data = []

    for metric_type, metric in tasks:
        print(f"Running [{metric_type}] {metric}...", end=" ", flush=True)

        # TODO: Implement the different metric calculations


if __name__ == "__main__":
    main()
