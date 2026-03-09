# std imports
import sys
import json
import logging
import argparse
from pathlib import Path

# 3rd party imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

# local imports
from shared_utils.custom_logging import create_logger

# --- GLOBAL DEFINITIONS ---

# --- create global logger ---
if __debug__:
    logger = create_logger("EnsembleEvaluationLogger", log_level=logging.DEBUG)
    logger.newline()
    logger.debug('--- DEBUG ON -- (Log level: DEBUG)')
    logger.newline()
else:
    logger = create_logger("EnsembleEvaluationLogger", log_level=logging.INFO)
    logger.newline()
    logger.info('--- DEBUG OFF --- (Log level: INFO)')
    logger.newline()

# --- metrics ---
CLASS_WISE_METRICS = [
        'Classification Report',
]

GLOBAL_METRICS = [
        'Cohen`s Kappa', 
        'Matthews Correlation Coefficient', 
]

VISUALIZATIONS = [
        'Heatmap', 
        'One-vs-Rest ROC',
        'One-vs-One ROC', 
        'Macro-averaged ROC', 
        'One-vs-Rest Precision-Recall Curve',
]

ALL_METRICS = CLASS_WISE_METRICS + GLOBAL_METRICS + VISUALIZATIONS

# --- HELPER FUNCTIONS ---

def define_params():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble Inference Results')
    
    # data arguments
    parser.add_argument('--results', type=str, required=True, help='Path to tensor with inference results (M, N, C)')
    parser.add_argument('--targets', type=str, required=True, help='Path to tensor with targets (N,)')
    parser.add_argument('--member_ids', type=int, nargs='+', help='List of ensemble members (IDs) to perform evaluation on')
    parser.add_argument('--outdir', type=str, default='.', help='Directory for output file(s)')

    # metric arguments
    parser.add_argument('--metric-types', type=str, default='all', choices=['all', 'class-wise', 'global', 'visualization'], nargs="+", help='Select metric type(s) to evaluate with. Defaults to "all"')
    parser.add_argument('--metrics', type=str, default=[], choices=ALL_METRICS, nargs="+", help='Evaluate using specific metric(s), overrides --metric-types')
    parser.add_argument('--normalizeheatmap', type=str, default=None, choices=['true', 'pred', 'all'], help='Normalize the heatmap by row, column or all values. Defaults to None.')
    
    # seed
    parser.add_argument('--seed', type=int, default=666, help='Random seed')

    return parser.parse_args()

def load_data(results_path, targets_path, indices):
    """Loads tensors and infers dimensions M, N, C."""
    logger.info(f"\nLoading data from:\n  Probas: {results_path}\n  Targets: {targets_path}")
    
    probas = torch.load(Path(results_path).resolve(), map_location='cpu').float().numpy()
    targets = torch.load(Path(targets_path).resolve(), map_location='cpu').long().numpy()

    # filter unneeded indices
    probas = probas[indices]

    M, N, C = probas.shape

    logger.info(f"Inferred Dimensions -> Models (M): {M}, Samples (N): {N}, Classes (C): {C}")
    return probas, targets, M, N, C

def ensemble_decision(probas):
    # derive the ensemble_decision based on soft voting
    # average probas across all M models -> (N, C)
    avg_probas = np.mean(probas, axis=0)
    # argmax to get class predictions -> (N, )
    y_pred = np.argmax(avg_probas, axis=1)
    return y_pred, avg_probas

# --- MAIN LOGIC ---
def main():
    args = define_params()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # load data
    try:
        probas, targets, M, N, C = load_data(args.results, args.targets, args.member_ids)
    except Exception as e:
        logger.critical(f'Error loading data: {e}')
        sys.exit(1)

    # derive decisions
    y_pred, y_score = ensemble_decision(probas)

    # determine execution list
    tasks = [] # list of tuples (Metric-Type, Metric)

    # if specific metrics are requested, find them
    if args.metrics:
        not_found = []
        for idx, metric in enumerate(args.metrics):
            if metric in GLOBAL_METRICS:
                tasks.append(("global", metric))
            elif metric in CLASS_WISE_METRICS:
                tasks.append(("class-wise", metric))
            elif metric in VISUALIZATIONS:
                tasks.append(("visualization", metric))
            else:
                not_found.append(metric)
        if not_found:
            logger.newline()
            logger.warning(f'Warning: The following metrics could not be found: {not_found}. Continuing without these metrics.')
            logger.newline()
    else:
        # otherwise append all metrics from metric types
        metric_types = set(args.metric_types) if isinstance(args.metric_types, list) else {args.metric_types}
        if metric_types & {'all', 'global'}:
            for metric in GLOBAL_METRICS:
                tasks.append(('global', metric))
        if metric_types & {'all', 'class-wise'}:
            for metric in CLASS_WISE_METRICS: 
                tasks.append(('class-wise', metric))
        if metric_types & {'all', 'visualization'}:
            for metric in VISUALIZATIONS:
                tasks.append(('visualization', metric))

    global_metric_results = {}
    class_wise_metric_results = {}
    generated_plots = []

    for metric_type, metric in tasks:
        logger.info(f"Running [{metric_type}] {metric}...")

        try:
            if metric_type == 'global':
                if metric == 'Cohen`s Kappa':
                    global_metric_results[metric] = cohen_kappa_score(targets, y_pred)
                elif metric == 'Matthews Correlation Coefficient':
                    global_metric_results[metric] = matthews_corrcoef(targets, y_pred)
                logger.info(f"{metric} ({metric_type}) done: {global_metric_results[metric]:.4f}")

            elif metric_type == 'class-wise':
                report = classification_report(targets, y_pred, output_dict=True, zero_division=0)
                report['accuracy'] = {k: (report['accuracy'] if k != 'support' else report['macro avg']['support']) for k in report['macro avg']}
                class_wise_metric_results[metric] = report
                classification_report_str = classification_report(targets, y_pred, zero_division=0)
                logger.info(f"{metric} ({metric_type}) done:\n{classification_report_str}")

            elif metric_type == 'visualization':
                plt.figure(figsize=(12, 10))
                filename = f"{metric.replace(' ', '_').lower()}.png"
                
                if metric == 'Heatmap':
                    cm = confusion_matrix(targets, y_pred, normalize=args.normalizeheatmap)
                    sns.heatmap(cm, annot=False, cmap='Blues')
                    plt.title(metric)
                elif 'ROC' in metric: 
                    y_bin = label_binarize(targets, classes=np.arange(C))
                    colors = plt.get_cmap('tab20', C)
                    for i in range(C):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                        plt.plot(fpr, tpr, label=f'Class {i}', color=colors(i))
                    plt.legend(loc='best', fontsize='xx-small', ncol=2)
                    plt.title(metric)
                elif metric == 'One-vs-Rest Precision-Recall Curve':
                    y_bin = label_binarize(targets, classes=np.arange(C))
                    colors = plt.get_cmap('tab20', C)
                    for i in range(C):
                        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                        plt.plot(rec, prec, label=f'Class {i}', color=colors(i))
                    plt.legend(loc='best', fontsize='xx-small', ncol=2)
                    plt.title(metric)
                
                plt.savefig(outdir / filename)
                plt.close()
                generated_plots.append(filename)
                logger.info(f"Saved Plot for metric: {metric}")

        except Exception as e:
            logger.error(f"Failed processing metric {metric} of type {metric_type}: {e}")

    # export non-visual results to JSON
    with open(outdir / "metrics_results.json", "w") as f:
        all_data = global_metric_results.copy()
        all_data.update(class_wise_metric_results)
        json.dump(all_data, f, indent=4)
    
    # HTML export with embedded images
    html_content = f"""
    <html>
    <head>
        <title>Ensemble Evaluation Report</title>
    </head>
    <body>
        <h1>Ensemble Evaluation Results</h1>
        <p><strong>Evaluated Results:</strong> {args.results}</p>
        <p><strong>Using Targets:</strong> {args.targets}</p>
        <p><strong>Evaluated Ensemble Members:</strong> {args.member_ids}</p>
    """

    # global metrics table
    if global_metric_results:
        html_content += "<h1>Global Metrics</h1>"
        df_global = pd.DataFrame.from_dict(global_metric_results, orient='index', columns=['Value'])
        html_content += df_global.to_html()
            
    # class-wise report table
    if class_wise_metric_results:
        html_content += "<h1>Class-wise Metrics</h1>"
        df_class_wise = pd.DataFrame.from_dict(class_wise_metric_results['Classification Report']).transpose()
        html_content += df_class_wise.to_html()

    # visualizations
    if generated_plots:
        html_content += "<h1>Visualizations</h1>"
        for plot_file in generated_plots:
            html_content += f"<h3>{plot_file.replace('_', ' ').replace('.png', '').title()}</h3>"
            html_content += f'<img src="{plot_file}" width="800"><br>'

    html_content += "</body></html>"
    
    with open(outdir / "report.html", "w") as f:
        f.write(html_content)

    logger.info(f"Results and visual report saved to {outdir}")

if __name__ == "__main__":
    main()
