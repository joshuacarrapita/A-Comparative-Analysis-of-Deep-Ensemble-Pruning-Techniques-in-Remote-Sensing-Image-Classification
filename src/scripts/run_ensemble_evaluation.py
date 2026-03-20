# std imports
import os
import sys
import json
import time
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
    classification_report, roc_auc_score, auc, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

# local imports
from shared_utils.custom_logging import create_logger
from configilm.extra.BENv2_utils import NEW_LABELS, NEW_LABELS_ORIGINAL_ORDER, NEW_LABELS_16, NEW_LABELS_16_ORIGINAL_ORDER

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
        'Classification Report', # checked
        # TODO: Add One-vs-Rest ROC AUC's
]

GLOBAL_METRICS = [
        'Cohen`s Kappa', # DONE: check
        'Matthews Correlation Coefficient', # DONE: check
        'Micro-Averaged One-vs-Rest ROC AUC', # DONE: check
        'Macro-Averaged One-vs-Rest ROC AUC', # DONE: check
        # TODO: check and maybe add
        # Average Precision
        # Brier Score
        # ECE

]

VISUALIZATIONS = [ # TODO: check
        'Heatmap',
        'One-vs-Rest ROC Curves', # DONE: check
            # 'One-vs-One ROC', # DONE: excluded because it would be too much
        'Macro-Averaged One-vs-Rest ROC Curve',  # DONE: check
        'Micro-Averaged One-vs-Rest ROC Curve',  # DONE: check
        'All-in-One One-vs-Rest ROC Curves', # DONE: check
        'One-vs-Rest Precision-Recall Curve',
]

ALL_METRICS = CLASS_WISE_METRICS + GLOBAL_METRICS + VISUALIZATIONS

# --- HELPER FUNCTIONS ---

def define_params():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble Inference Results')
    
    # data arguments
    parser.add_argument('--inference_results', type=Path, required=True, help='Path to tensor with inference results (M, N, C)')
    parser.add_argument('--targets', type=Path, required=True, help='Path to tensor with targets (N,)')
    parser.add_argument('--pruning_results_root', type=Path, default='', help='Path to directory or .csv with (separate!) pruning results (M, N, C)')
    parser.add_argument('--member_ids', type=int, nargs='+', help='List of ensemble members (IDs) to perform evaluation on')
    parser.add_argument('--outdir', type=str, default='.', help='Directory for output file(s)')

    # metric arguments
    parser.add_argument('--metric-types', type=str, default='all', choices=['all', 'class-wise', 'global', 'visualization'], nargs="+", help='Select metric type(s) to evaluate with. Defaults to "all"')
    parser.add_argument('--metrics', type=str, default=[], choices=ALL_METRICS, nargs="+", help='Evaluate using specific metric(s), overrides --metric-types')
    parser.add_argument('--normalizeheatmap', type=str, default=None, choices=['true', 'pred', 'all'], help='Normalize the heatmap by row, column or all values. Defaults to None.')
    
    # seed
    parser.add_argument('--seed', type=int, default=666, help='Random seed')

    return parser.parse_args()

def load_data(inference_results_path, targets_path):
    """Loads tensors and infers dimensions M, N, C."""
    logger.info(f"\nLoading data from:\n  Probas: {inference_results_path}\n  Targets: {targets_path}")
    
    probas = torch.load(Path(inference_results_path).resolve(), map_location='cpu').float().numpy()
    targets = torch.load(Path(targets_path).resolve(), map_location='cpu').long().numpy()

    M, N, C = probas.shape

    logger.info(f"Inferred Dimensions of Loaded Data -> Models (M): {M}, Samples (N): {N}, Classes (C): {C}")
    return probas, targets, M, N, C

def filter_data(probas, member_ids):
    # filter unneeded members
    probas = probas[member_ids]

    M, N, C = probas.shape

    logger.info(f"Inferred Dimensions after Filtering -> Models (M): {M}, Samples (N): {N}, Classes (C): {C}")
    return probas, M 

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
    slurm_job_id = os.getenv('SLURM_JOB_ID', default='')
    slurm_array_job_id = os.getenv('SLURM_ARRAY_JOB_ID', default='')
    start_time = ('' if slurm_job_id != '' else time.strftime("%d-%m-%Y_%H-%M-%S"))
    outdir = Path(args.outdir).resolve() / 'evaluation' / f'{slurm_array_job_id}' / f'{start_time}'
    outdir.mkdir(parents=True, exist_ok=True)

    # load data
    try:
        probas, targets, M, N, C = load_data(args.inference_results, args.targets)
        targets_onehot = label_binarize(targets, classes=np.arange(C))
        LABELS = [f'Class {i}' for i in range(C)]
        if C == 16:
            for i in range(C):
                LABELS[i] = LABELS[i] + " - " + NEW_LABELS_16[i]
        elif C == 19:
            for i in range(C):
                LABELS[i] = LABELS[i] + " - " + NEW_LABELS[i]
            
    except Exception as e:
        logger.critical(f'Error loading data: {e}')
        sys.exit(1)
    
    # define evaluation space
    subensembles = pd.DataFrame(columns=['Library','Algorithm','Runtime','Result'])

    # evaluate subensemble if member IDs were specifically given
    if args.member_ids:
        member_ids_subensemble = pd.DataFrame([['manual','by_member_id',None,str(args.member_ids)]],columns=subensembles.columns)
        subensembles = pd.concat([subensembles,member_ids_subensemble], ignore_index=True)

    # evaluate subensembles found in pruning results files if any were given
    pruning_results_root = Path(args.pruning_results_root).resolve()
    pruning_results_file_paths = [pruning_results_root] if pruning_results_root.is_file() else list(pruning_results_root.glob(f'**/results.csv'))

    for file_path in pruning_results_file_paths:

        subensembles = pd.concat([subensembles,pd.read_csv(file_path)],ignore_index=True)

    # if neither of the two were given, evaluate full ensemble
    if subensembles.empty:
        full_ensemble = pd.DataFrame([['','',None,[i for i in range(M)]]],columns=subensembles.columns)
        subensembles = pd.concat([subensembles,full_ensemble], ignore_index=True)
        full_ensemble = True
    else:
        full_ensemble = False

    for idx, subensemble in subensembles.iterrows():
        subensemble['Result'] = eval(subensemble['Result'])
        
        # adjust path if evaluating subensembles
        if full_ensemble:
            outdir_run = outdir
        else: 
            outdir_run = outdir / f'{len(subensemble["Result"])}' / f'{subensemble["Library"]}' / f'{subensemble["Algorithm"]}' / f'{"-".join(str(i) for i in subensemble["Result"])}' / f'{slurm_job_id}' / f'{idx}'

        # filter data
        try:
            print(type(subensemble['Result']))
            probas_filtered, M = filter_data(probas, subensemble['Result'])
        except Exception as e:
            logger.critical(f'Error filtering data: {e}')
            sys.exit(1)

        # derive decisions
        y_pred, y_score = ensemble_decision(probas_filtered)

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
                        global_metric_results[metric] = {'': matthews_corrcoef(targets, y_pred)}
                        logger.info(f"{metric} ({metric_type}) done: {global_metric_results[metric]['']:.4f}")
                    elif metric == 'Micro-Averaged One-vs-Rest ROC AUC':
                        micro_roc_auc_ovr = roc_auc_score(targets_onehot, y_score, multi_class="ovr", average="micro",)
                        logger.info(f"{metric} ({metric_type}) done: {micro_roc_auc_ovr:.4f}")
                        if np.isnan(micro_roc_auc_ovr).any(): micro_roc_auc_ovr = 'N/A'
                        global_metric_results[metric] = {'': micro_roc_auc_ovr}
                    elif metric == 'Macro-Averaged One-vs-Rest ROC AUC':
                        macro_roc_auc_ovr = roc_auc_score(targets_onehot, y_score, multi_class="ovr", average="macro",)
                        logger.info(f"{metric} ({metric_type}) done: {macro_roc_auc_ovr:.4f}")
                        if np.isnan(macro_roc_auc_ovr).any(): macro_roc_auc_ovr = 'N/A'
                        global_metric_results[metric] = {'': macro_roc_auc_ovr}

                elif metric_type == 'class-wise':
                    report = classification_report(targets, y_pred, output_dict=True, zero_division=0)
                    report['accuracy'] = {k: (report['accuracy'] if k != 'support' else report['macro avg']['support']) for k in report['macro avg']}
                    class_wise_metric_results[metric] = report
                    classification_report_str = classification_report(targets, y_pred, zero_division=0)
                    logger.info(f"{metric} ({metric_type}) done:\n{classification_report_str}")

                elif metric_type == 'visualization':
                    plt.figure(figsize=(12, 10))
                    filename = f"{metric.replace(' ', '_').lower()}.png"
                    empty_figure = True

                    if metric == 'Heatmap':
                        empty_figure = False
                        cm = confusion_matrix(targets, y_pred, normalize=args.normalizeheatmap)
                        sns.heatmap(cm, annot=False, cmap='Blues')
                        plt.title(metric)
                    if metric == 'One-vs-Rest ROC Curves' or metric == 'All-in-One One-vs-Rest ROC Curves':
                        colors = plt.get_cmap('tab20', C)
                        for i in range(C):
                            fpr, tpr, _ = roc_curve(targets_onehot[:, i], y_score[:, i])
                            if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                                empty_figure = False
                                roc_auc = auc(fpr, tpr)
                                plt.plot(fpr, tpr, label=f'{LABELS[i]} (AUC = {roc_auc:.2f})', color=colors(i),)
                        plt.legend(loc='best', fontsize='xx-small', ncol=1)
                        plt.title(metric)
                    if metric == 'Micro-Averaged One-vs-Rest ROC Curve' or metric == 'All-in-One One-vs-Rest ROC Curves':
                        fpr, tpr, _ = roc_curve(targets_onehot.ravel(), y_score.ravel())
                        if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                            empty_figure = False
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, label=f'Micro-Averaged (AUC = {roc_auc:.2f})', color="deeppink",)
                        plt.legend(loc='best', fontsize='xx-small', ncol=1)
                        plt.title(metric)
                    if metric == 'Macro-Averaged One-vs-Rest ROC Curve' or metric == 'All-in-One One-vs-Rest ROC Curves':
                        fpr_grid = np.linspace(0.0,1.0,1000)
                        mean_tpr = np.zeros_like(fpr_grid)
                        feasible = True
                        for i in range(C):
                            fpr, tpr, _ = roc_curve(targets_onehot[:, i], y_score[:, i])
                            if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                                mean_tpr += np.interp(fpr_grid, fpr, tpr)
                            else:
                                feasible = False
                        if feasible == True:
                            empty_figure = False
                            mean_tpr /= C
                            fpr = fpr_grid
                            tpr = mean_tpr
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, label=f'Macro-Averaged (AUC = {roc_auc:.2f})', color="navy",)
                            plt.legend(loc='best', fontsize='xx-small', ncol=1)
                            plt.title(metric)
                    if metric == 'All-in-One One-vs-Rest ROC Curves':
                        plt.legend(loc='best', fontsize='xx-small', ncol=1)
                        plt.title(metric)
                    if metric == 'One-vs-Rest Precision-Recall Curve':
                        empty_figure = False # TODO: Figure out if there are cases where PR-Curve is N/A
                        y_bin = label_binarize(targets, classes=np.arange(C))
                        colors = plt.get_cmap('tab20', C)
                        for i in range(C):
                            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                            plt.plot(rec, prec, label=f'{LABELS[i]}', color=colors(i))
                        plt.legend(loc='best', fontsize='xx-small', ncol=1)
                        plt.title(metric)
                    
                    resources = outdir_run / 'resources'
                    resources.mkdir(exist_ok=True, parents=True)
                    if not empty_figure:
                        plt.savefig(resources / filename)
                        generated_plots.append(filename)
                        logger.info(f"Saved Plot for metric: {metric}")
                    else:
                        logger.info(f"Did not save plot for metric: {metric} due to inapplicability")
                    plt.close()

            except Exception as e:
                logger.error(f"Failed processing metric {metric} of type {metric_type}: {e}")

        # export non-visual results to JSON
        with open(outdir_run / "metrics_results.json", "w") as f:
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
            <p><strong>Evaluated Results:</strong> {args.inference_results}</p>
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
                html_content += f'<img src="resources/{plot_file}" width="800"><br>'

        html_content += "</body></html>"
        
        with open(outdir_run / "report.html", "w") as f:
            f.write(html_content)

        logger.info(f"Results and visual report saved to {outdir_run}")

if __name__ == "__main__":
    main()
