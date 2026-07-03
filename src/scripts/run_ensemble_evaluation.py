# std imports
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from glob import glob as lnsglob

# 3rd party imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report, roc_auc_score, auc, roc_curve, 
    precision_recall_curve, average_precision_score, brier_score_loss,
)
from sklearn.preprocessing import label_binarize
from alive_progress import alive_it

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
CLASS_WISE_METRICS = [ # DONE: check
        'Classification Report', # DONE: check
        'Per-Class One-vs-Rest ROC AUC', # DONE: check
        'Per-Class One-vs-Rest PR AUC', # DONE: check
        'Per-Class Brier Score Loss', # DONE: check
]

GLOBAL_METRICS = [ # DONE: check
        'Cohen`s Kappa', # DONE: check
        'Matthews Correlation Coefficient', # DONE: check
        'Micro-Average One-vs-Rest ROC AUC', # DONE: check
        'Macro-Average One-vs-Rest ROC AUC', # DONE: check
        'Weighted-Average One-vs-Rest ROC AUC', # DONE: check
        'Micro-Average One-vs-Rest PR AUC', # DONE: check
        'Macro-Average One-vs-Rest PR AUC', # DONE: check
        'Weighted-Average One-vs-Rest PR AUC', # DONE: check
        'Brier Score Loss', # DONE: check
        'Expected Calibration Error', # DONE: check
]

VISUALIZATIONS = [ # DONE: check
        'Heatmap', # DONE: check
        'Per-Class One-vs-Rest ROC Curves', # DONE: check
#       'One-vs-One ROC Curves', # DONE: excluded because it would be too much
        'Micro-Average One-vs-Rest ROC Curve',  # DONE: check
        'Macro-Average One-vs-Rest ROC Curve',  # DONE: check
#       'Weighted-Average One-vs-Rest ROC Curve',  # DONE: excluded because it would be too much
        'Per-Class One-vs-Rest PR Curves', # DONE: check
#       'One-vs-One PR Curves', # DONE: excluded because it would be too much
        'Micro-Average One-vs-Rest PR Curve',  # DONE: check
        'Macro-Average One-vs-Rest PR Curve',  # DONE: check
#       'Weighted-Average One-vs-Rest PR Curve',  # DONE: excluded because it would be too much
        'All-in-One One-vs-Rest PR Curves', # DONE: check
]

ALL_METRICS = CLASS_WISE_METRICS + GLOBAL_METRICS + VISUALIZATIONS

VALIDATION_TO_TEST = None

# --- HELPER FUNCTIONS ---

def define_params():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble Inference Results')
    
    # data arguments
    parser.add_argument('--inference_results', type=Path, required=True, help='Path to tensor with inference results (M, N, C)')
    parser.add_argument('--targets', type=Path, required=True, help='Path to tensor with targets (N,)')
    parser.add_argument('--validation_test_mapping', type=Path, required=False, help='Path to json that maps model indices of validation set inference to those of test set inference')
    parser.add_argument('--pruning_results_root', type=str, default='', help='Path to directory or .csv with (separate!) pruning results (M, N, C)')
    parser.add_argument('--member_ids', type=int, nargs='+', help='List of ensemble members (IDs) to perform evaluation on')
    parser.add_argument('--outdir', type=str, default='.', help='Directory for output file(s)')

    # metric arguments
    parser.add_argument('--metric-types', type=str, default='all', choices=['all', 'class-wise', 'global', 'visualization'], nargs="+", help='Select metric type(s) to evaluate with. Defaults to "all"')
    parser.add_argument('--metrics', type=str, default=[], choices=ALL_METRICS, nargs="+", help='Evaluate using specific metric(s), overrides --metric-types')
    parser.add_argument('--normalizeheatmap', type=str, default='true', choices=['None', 'true', 'pred', 'all'], help='Normalize the heatmap by row, column or all values. Defaults to "true".')
    
    # seed
    parser.add_argument('--seed', type=int, default=666, help='Random seed')

    args = parser.parse_args()
    
    if args.normalizeheatmap == 'None': args.normalizeheatmap = None

    return args

def load_data(inference_results_path, targets_path, validation_test_mapping):
    """Loads tensors and mappings and infers dimensions M, N, C."""
    logger.info(f"\nLoading data from:\n  Probas: {inference_results_path}\n  Targets: {targets_path}")
    
    probas = torch.load(Path(inference_results_path).resolve(), map_location='cpu').float().numpy()
    targets = torch.load(Path(targets_path).resolve(), map_location='cpu').long().numpy()

    M, N, C = probas.shape

    logger.info(f"Inferred Dimensions of Loaded Data -> Models (M): {M}, Samples (N): {N}, Classes (C): {C}")

    if validation_test_mapping:
        validation_test_mapping = Path(validation_test_mapping).resolve()
        with open(validation_test_mapping, 'r') as f:
            global VALIDATION_TO_TEST
            VALIDATION_TO_TEST = json.loads(f.read())

    return probas, targets, M, N, C

def filter_data(probas, member_ids):
    # filter unneeded members
    probas = probas[member_ids]

    M, N, C = probas.shape

    logger.info(f"Inferred Dimensions after Filtering -> Models (M): {M}, Samples (N): {N}, Classes (C): {C}")
    return probas, M 

def ensemble_decision(probas):
    M, N, C = probas.shape

    # derive the ensemble_decision based on soft voting
    # average probas across all M models -> (N, C)
    avg_probas = np.mean(probas, axis=0)
    # argmax to get class predictions -> (N, )
    y_pred = np.argmax(avg_probas, axis=1)
    return y_pred, avg_probas

def my_cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates the Cohen Kappa Score manually in order to also yield observed
    accuracy (p_o) and expected accuracy (p_e) from the formula 
    k = (p_o - p_e) / (1 - p_e). 

    k is a number between -1 and 1. The maximum value means complete agreement; zero or lower means chance agreement.
    """
    cm = confusion_matrix(y_true, y_pred)
    total_samples = np.sum(cm)

    # p_o = correct predictions / total samples
    p_o = np.trace(cm) / total_samples

    sum_cols = np.sum(cm, axis=0)
    sum_rows = np.sum(cm, axis=1)
    p_e = np.sum((sum_rows * sum_cols) / total_samples) / total_samples

    k = (p_o - p_e) / (1 - p_e) if p_e != 1 else numpy.nan

    accuracy_gain = k * (1 - p_e)

    sklearn_k = cohen_kappa_score(y_true, y_pred)
    matches_sklearn = np.isclose(k,sklearn_k)

    return {
        'k': k,
        'Expected Accuracy': p_e,
        'Observed Accuracy': p_o,
        'Accuracy Gain<br>(over "Expected Accuracy")': accuracy_gain,
        'Matches Sklearn': bool(matches_sklearn),
    }

def expected_calibration_error(samples, true_labels, M=10):
    # adapted from https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece

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
        probas, targets, M, N, C = load_data(args.inference_results, args.targets, args.validation_test_mapping)
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
    
    logger.info("Collecting subensemble info...")

    # define evaluation space
    subensembles = pd.DataFrame(columns=['Library','Algorithm','Runtime','Result'])

    # evaluate subensemble if member IDs were specifically given
    if args.member_ids:
        logger.info("Collecting subensemble info from member_ids...")
        member_ids_subensemble = pd.DataFrame([['manual','by_member_id',None,str(args.member_ids)]],columns=subensembles.columns)
        subensembles = pd.concat([subensembles,member_ids_subensemble], ignore_index=True)

    # evaluate subensembles found in pruning results files if any were given
    logger.info("Collecting subensemble info from pruning_results_root...")
    if args.pruning_results_root:
        pruning_results_root = Path(args.pruning_results_root).resolve() 
        pruning_results_file_paths = [pruning_results_root] if pruning_results_root.is_file() else list(lnsglob(f'{pruning_results_root}/**/results.csv', recursive=True))
        for file_path in alive_it(pruning_results_file_paths):

            subensembles = pd.concat([subensembles,pd.read_csv(file_path)],ignore_index=True)

    # if neither of the two were given, evaluate full ensemble
    if subensembles.empty:
        logger.info("Adding full ensemble for evaluation...")
        full_ensemble = pd.DataFrame([['','',None,str([i for i in range(M)])]],columns=subensembles.columns)
        subensembles = pd.concat([subensembles,full_ensemble], ignore_index=True)
        full_ensemble = True
    else:
        full_ensemble = False

    logger.info("Evaluating subensembles...")

    for idx, subensemble in alive_it(subensembles.iterrows(), total=len(subensembles.index)):
        logger.info(f"Evaluating {subensemble['Library']}, {subensemble['Algorithm']} result: {subensemble['Result']}...")
        try:
            if VALIDATION_TO_TEST:
                logger.info("Using validation -> test mapping...")
                logger.debug(VALIDATION_TO_TEST)
                subensemble['Result'] = [VALIDATION_TO_TEST[str(x)] for x in eval(subensemble['Result'])]
            else:
                logger.info("NOT USING validation -> test mapping!")
                subensemble['Result'] = [x for x in eval(subensemble['Result'])]
        except Exception as e:
            logger.error(f"Encountered error during subensemble parsing: {e}")
            logger.warning(f"Skipping {subensemble['Library']}, {subensemble['Algorithm']} result: {subensemble['Result']}...")
            continue
        
        # adjust path if evaluating subensembles
        if full_ensemble:
            outdir_run = outdir
        else: 
            outdir_run = outdir / f'{len(subensemble["Result"])}' / f'{subensemble["Library"]}' / f'{subensemble["Algorithm"]}' / f'{slurm_job_id}' 
        outdir_run.mkdir(parents=True, exist_ok=True)

        # filter data
        try:
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
                        global_metric_results[metric] = my_cohen_kappa_score(targets, y_pred)

                    elif metric == 'Matthews Correlation Coefficient':
                        global_metric_results[metric] = {'': matthews_corrcoef(targets, y_pred)}

                    elif metric == 'Micro-Average One-vs-Rest ROC AUC':
                        micro_roc_auc_ovr = roc_auc_score(targets_onehot, y_score, multi_class="ovr", average="micro")
                        if np.isnan(micro_roc_auc_ovr).any(): micro_roc_auc_ovr = 'NaN'
                        if not "Average One-vs-Rest ROC AUC" in global_metric_results:
                            global_metric_results['Average One-vs-Rest ROC AUC'] = {f"Micro": micro_roc_auc_ovr}
                        else:
                            global_metric_results['Average One-vs-Rest ROC AUC'].update({f"Micro": micro_roc_auc_ovr})

                    elif metric == 'Macro-Average One-vs-Rest ROC AUC':
                        macro_roc_auc_ovr = roc_auc_score(targets_onehot, y_score, multi_class="ovr", average="macro")
                        if np.isnan(macro_roc_auc_ovr).any(): macro_roc_auc_ovr = 'NaN'
                        if not "Average One-vs-Rest ROC AUC" in global_metric_results:
                            global_metric_results['Average One-vs-Rest ROC AUC'] = {f"Macro": macro_roc_auc_ovr}
                        else:
                            global_metric_results['Average One-vs-Rest ROC AUC'].update({f"Macro": macro_roc_auc_ovr})

                    elif metric == 'Weighted-Average One-vs-Rest ROC AUC':
                        weighted_roc_auc_ovr = roc_auc_score(targets_onehot, y_score, multi_class="ovr", average="weighted")
                        if np.isnan(weighted_roc_auc_ovr).any(): weighted_roc_auc_ovr = 'NaN'
                        if not "Average One-vs-Rest ROC AUC" in global_metric_results:
                            global_metric_results['Average One-vs-Rest ROC AUC'] = {f"Weighted": weighted_roc_auc_ovr}
                        else:
                            global_metric_results['Average One-vs-Rest ROC AUC'].update({f"Weighted": weighted_roc_auc_ovr})

                    elif metric == 'Micro-Average One-vs-Rest PR AUC':
                        micro_ovr_pr_auc = average_precision_score(targets_onehot, y_score, average='micro')
                        if np.isnan(micro_ovr_pr_auc).any(): micro_ovr_pr_auc = 'NaN'
                        if not 'Average One-vs-Rest PR AUC<br><br>(Average Precision Score)' in global_metric_results:
                            global_metric_results['Average One-vs-Rest PR AUC<br><br>(Average Precision Score)'] = {f'Micro': micro_ovr_pr_auc}
                        else: 
                            global_metric_results['Average One-vs-Rest PR AUC<br><br>(Average Precision Score)'].update({f'Micro': micro_ovr_pr_auc})

                    elif metric == 'Macro-Average One-vs-Rest PR AUC':
                        macro_ovr_pr_auc = average_precision_score(targets_onehot, y_score, average='macro')
                        if np.isnan(macro_ovr_pr_auc).any(): macro_ovr_pr_auc = 'NaN'
                        if not 'Average One-vs-Rest PR AUC<br><br>(Average Precision Score)' in global_metric_results:
                            global_metric_results['Average One-vs-Rest PR AUC<br><br>(Average Precision Score)'] = {f'Macro': macro_ovr_pr_auc}
                        else: 
                            global_metric_results['Average One-vs-Rest PR AUC<br><br>(Average Precision Score)'].update({f'Macro': macro_ovr_pr_auc})

                    elif metric == 'Weighted-Average One-vs-Rest PR AUC':
                        weighted_ovr_pr_auc = average_precision_score(targets_onehot, y_score, average='weighted')
                        if np.isnan(weighted_ovr_pr_auc).any(): weighted_ovr_pr_auc = 'NaN'
                        if not 'Average One-vs-Rest PR AUC<br><br>(Average Precision Score)' in global_metric_results:
                            global_metric_results['Average One-vs-Rest PR AUC<br><br>(Average Precision Score)'] = {f'Weighted': weighted_ovr_pr_auc}
                        else: 
                            global_metric_results['Average One-vs-Rest PR AUC<br><br>(Average Precision Score)'].update({f'Weighted': weighted_ovr_pr_auc})

                    elif metric == 'Brier Score Loss':
                        micro_brier = brier_score_loss(targets, y_score) # standard brier
                        
                        class_briers = [brier_score_loss(targets_onehot[:, i], y_score[:, i], scale_by_half=False) for i in range(C)]
                        
                        macro_brier = np.mean(class_briers)
                        
                        supports = np.sum(targets_onehot, axis=0)
                        weighted_brier = np.average(class_briers, weights=supports)

                        global_metric_results[metric] = {
                            'Micro (Overall)': micro_brier,
                            'Macro': macro_brier,
                            'Weighted': weighted_brier
                        }

                    elif metric == 'Expected Calibration Error':
                        global_metric_results[metric] = {'': expected_calibration_error(y_score, targets)}

                    logger.info(f"[{metric_type}] {metric} done")

                elif metric_type == 'class-wise':

                    if metric == 'Classification Report':
                        report = classification_report(targets, y_pred, output_dict=True, zero_division=0)

                        # Move accuracy to global metrics as it is calculated over all samples
                        global_metric_results['Average Accuracy'] = {'Micro<br>(= Micro Precision, Recall, F1-Score)': report['accuracy']} # = sum_0^n-samples(tp) / (sum_0^n-samples(tp) + sum_0^n-samples(fp)) (or fn instead of fp, doesn't matter) = sum(correct) / (sum(correct) + sum(incorrect)) = sum(correct) / total_samples
                        report.pop('accuracy')
                        global_metric_results['Average Accuracy'].update({'Macro<br>(= Macro Recall)': report['macro avg']['recall']}) # = sum_0^n-classes(tp_i / (tp_i + fn_i)) = sum_0^n-classes(#correct_i / #total_i)

                        global_metric_results['Average Accuracy'].update({'Weighted<br>(= Weighted Recall)': report['weighted avg']['recall']})

                        # Move Macro and Weighteds to Global Metrics
                        global_metric_results['Average Precision'] = {'Macro': report['macro avg']['precision']}
                        global_metric_results['Average Recall'] = {'Macro': report['macro avg']['recall']}
                        global_metric_results['Average F1-Score'] = {'Macro': report['macro avg']['f1-score']}

                        global_metric_results['Average Precision'].update({'Weighted': report['weighted avg']['precision']})
                        global_metric_results['Average Recall'].update({'Weighted': report['weighted avg']['recall']})
                        global_metric_results['Average F1-Score'].update({'Weighted': report['weighted avg']['f1-score']})

                        report.pop('macro avg')
                        report.pop('weighted avg')

                        # Move Precision, Recall and F1 averages to global as they are calculated over all samples
                        class_wise_metric_results[metric] = report
                        classification_report_str = classification_report(targets, y_pred, zero_division=0)

                    elif metric == 'Per-Class One-vs-Rest ROC AUC':
                        for i in range(C):
                            fpr, tpr, _ = roc_curve(targets_onehot[:, i], y_score[:, i])
                            if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                                roc_auc = auc(fpr, tpr)
                                class_wise_metric_results["Classification Report"][f"{i}"]["ROC AUC"] = roc_auc

                    elif metric == 'Per-Class One-vs-Rest PR AUC':
                        pc_ovr_pr_auc = average_precision_score(targets_onehot, y_score, average=None)
                        for i in range(C):
                            class_wise_metric_results["Classification Report"][f"{i}"]["PR AUC"] = pc_ovr_pr_auc[i]

                    elif metric == 'Per-Class Brier Score Loss':
                        for i in range(C):
                            brier = brier_score_loss(targets_onehot[:, i], y_score[:, i], scale_by_half=False)
                            class_wise_metric_results["Classification Report"][f"{i}"]["Brier Score"] = brier

                    logger.info(f"[{metric_type}] {metric} done")

                elif metric_type == 'visualization':
                    colors = plt.get_cmap('tab20', C)
                    plt.figure(figsize=(12, 10))
                    filename = f"{metric.replace(' ', '_').lower()}.png"
                    empty_figure = True

                    if metric == 'Heatmap':

                        empty_figure = False

                        fig, axes = plt.subplots(1, 2, width_ratios=[5,1], figsize=(18,10))
                        fig.tight_layout(pad=2)

                        cm = confusion_matrix(targets, y_pred, normalize=args.normalizeheatmap)
                        sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[0])
                        axes[0].set_title(metric, fontsize='xx-large')
                        axes[0].set_xlabel("Predicted", fontsize='x-large')
                        axes[0].set_ylabel("True", fontsize='x-large')

                        axes[1].set_axis_off()
                        axes[1].set_title('Legend', fontsize='xx-large')
                        axes[1].text(-0.2, 0, s='\n'.join(LABELS), fontsize='xx-large', wrap=True, bbox=dict(boxstyle='round', facecolor='white')) 

                    if metric == 'Per-Class One-vs-Rest ROC Curves':
                        resources = outdir_run / 'resources'
                        resources.mkdir(exist_ok=True, parents=True)
                        
                        for i in range(C):
                            fpr, tpr, _ = roc_curve(targets_onehot[:, i], y_score[:, i])
                            if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                                plt.figure(figsize=(12, 10))
                                roc_auc = auc(fpr, tpr)
                                plt.plot(fpr, tpr, label=f'{LABELS[i]} (AUC = {roc_auc:.2f})', color=colors(i))
                                
                                plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Baseline')
                                
                                plt.xlabel('FPR', fontsize='x-large')
                                plt.ylabel('TPR', fontsize='x-large')
                                plt.title(f'ROC Curve - {LABELS[i]}', fontsize='x-large')
                                plt.legend(loc='best', fontsize='medium')
                                
                                filename = f"roc_curve_class_{i}.png"
                                plt.savefig(resources / filename)
                                generated_plots.append(filename)
                                plt.close()

                        logger.info(f"[{metric_type}] {metric} done")
                        continue
                        
                    if metric == 'Micro-Average One-vs-Rest ROC Curve' or metric == 'All-in-One One-vs-Rest ROC Curves':
                        fpr, tpr, _ = roc_curve(targets_onehot.ravel(), y_score.ravel())
                        if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                            empty_figure = False
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, label=f'Micro-Average (AUC = {roc_auc:.2f})', color="deeppink")
                            if not metric == 'All-in-One One-vs-Rest ROC Curves':
                                plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Baseline')
                            plt.legend(loc='best', fontsize='medium', ncol=1)
                            plt.title(metric, fontsize='x-large')

                    if metric == 'Macro-Average One-vs-Rest ROC Curve' or metric == 'All-in-One One-vs-Rest ROC Curves':
                        fpr_grid = np.linspace(0.0,1.0,1000)
                        mean_tpr = np.zeros_like(fpr_grid)
                        feasible = True
                        for i in range(C):
                            if feasible:
                                fpr, tpr, _ = roc_curve(targets_onehot[:, i], y_score[:, i])
                                if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                                    mean_tpr += np.interp(fpr_grid, fpr, tpr)
                                else:
                                    feasible = False
                        if feasible:
                            empty_figure = False
                            mean_tpr /= C
                            fpr = fpr_grid
                            tpr = mean_tpr
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, label=f'Macro-Average (AUC = {roc_auc:.2f})', color="navy")
                            if not metric == 'All-in-One One-vs-Rest ROC Curves':
                                plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Baseline')
                            plt.legend(loc='best', fontsize='medium', ncol=1)
                            plt.title(metric, fontsize='x-large')
                
                    if metric == 'All-in-One One-vs-Rest ROC Curves':
                        for i in range(C):
                            fpr, tpr, _ = roc_curve(targets_onehot[:, i], y_score[:, i])
                            if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                                empty_figure = False
                                roc_auc = auc(fpr, tpr)
                                plt.plot(fpr, tpr, label=f'{LABELS[i]} (AUC = {roc_auc:.2f})', color=colors(i))
                        plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Baseline')
                        plt.legend(loc='best', fontsize='medium', ncol=1)
                        plt.title(metric, fontsize='x-large')

                    if metric == 'Per-Class One-vs-Rest PR Curves':
                        resources = outdir_run / 'resources'
                        resources.mkdir(exist_ok=True, parents=True)
                        
                        for i in range(C):
                            prec, rec, _ = precision_recall_curve(targets_onehot[:, i], y_score[:, i])
                            if not np.isnan(prec).any() and not np.isnan(rec).any():
                                plt.figure(figsize=(12, 10))
                                plt.plot(rec, prec, label=f'{LABELS[i]}', color=colors(i))
                                
                                baseline = np.sum(targets_onehot[:, i]) / len(targets)
                                plt.axhline(y=baseline, linestyle='--', color='grey', label=f'Baseline ({baseline:.3f})')
                                
                                plt.xlabel('Recall', fontsize='x-large')
                                plt.ylabel('Precision', fontsize='x-large')
                                plt.title(f'PR Curve - {LABELS[i]}', fontsize='x-large')
                                plt.legend(loc='best', fontsize='medium')
                                
                                filename = f"pr_curve_class_{i}.png"
                                plt.savefig(resources / filename)
                                generated_plots.append(filename)
                                plt.close()

                        logger.info(f"[{metric_type}] {metric} done")
                        continue

                    if metric == 'Micro-Average One-vs-Rest PR Curve' or metric == 'All-in-One One-vs-Rest PR Curves':
                        prec, rec, _ = precision_recall_curve(targets_onehot.ravel(), y_score.ravel())
                        if not np.isnan(prec).any() and not np.isnan(rec).any():
                            empty_figure = False # TODO: Figure out if there are cases where PR Curve is N/A
                            micro_ovr_pr_auc = average_precision_score(targets_onehot, y_score, average='micro')
                            plt.plot(rec, prec, label=f'Micro-Average (AUC = {micro_ovr_pr_auc:.2f})', color='deeppink')

                            if not metric == 'All-in-One One-vs-Rest PR Curves':
                                baseline = 1 / C # Total Predictions = Total Samples $\times$ Total Classes ($N \times C$), Positive Labels = N ==> Baseline: $\frac{\text{Total Positives}}{\text{Total Predictions}} = \frac{N}{N \times C} = \frac{1}{C}$ 
                                plt.axhline(y=baseline, linestyle='--', color='grey', label=f'Baseline ({baseline:.3f})')
                                
                            plt.legend(loc='best', fontsize='medium', ncol=1)
                            plt.title(metric, fontsize='x-large')

                    if metric == 'Macro-Average One-vs-Rest PR Curve' or metric == 'All-in-One One-vs-Rest PR Curves':
                        rec_grid = np.linspace(0.0,1.0,1000)
                        macro_prec = np.zeros_like(rec_grid)
                        macro_ovr_pr_auc = 0
                        feasible = True
                        for i in range(C):
                            if feasible:
                                prec, rec, _ = precision_recall_curve(targets_onehot[:, i], y_score[:, i])
                                if not np.isnan(prec).any() and not np.isnan(rec).any():
                                    macro_prec += np.interp(rec_grid, np.flip(rec), np.flip(prec))
                                    macro_ovr_pr_auc += average_precision_score(targets_onehot[:, i], y_score[:, i])
                                else:
                                    feasible = False
                        if feasible:
                            empty_figure = False # TODO: Figure out if there are cases where PR Curve is N/A
                            macro_prec /= C
                            macro_ovr_pr_auc /= C
                            plt.plot(rec_grid, macro_prec, label=f'Macro-Average (AUC = {macro_ovr_pr_auc:.2f})', color='navy')
                            if not metric == 'All-in-One One-vs-Rest PR Curves':
                                baseline = 1 / C # $$\text{Macro Baseline} = \frac{1}{C} \sum_{i=1}^{C} \frac{N_i}{N}$$
                                plt.axhline(y=baseline, linestyle='--', color='grey', label=f'Baseline ({baseline:.3f})')
                                
                            plt.legend(loc='best', fontsize='medium', ncol=1)
                            plt.title(metric, fontsize='x-large')

                    if metric == 'All-in-One One-vs-Rest PR Curves':
                        for i in range(C):
                            prec, rec, _ = precision_recall_curve(targets_onehot[:, i], y_score[:, i])
                            if not np.isnan(prec).any() and not np.isnan(rec).any():
                                empty_figure = False # TODO: Figure out if there are cases where PR Curve is N/A
                                plt.plot(rec, prec, label=f'{LABELS[i]}', color=colors(i))
                        plt.legend(loc='best', fontsize='medium', ncol=1)
                        plt.title(metric, fontsize='x-large')

                    if 'ROC' in metric:
                        plt.xlabel('FPR',fontsize='x-large')
                        plt.ylabel('TPR',fontsize='x-large')
                    elif 'PR' in metric:
                        plt.xlabel('Recall',fontsize='x-large')
                        plt.ylabel('Precision',fontsize='x-large')

                    resources = outdir_run / 'resources'
                    resources.mkdir(exist_ok=True, parents=True)
                    if not empty_figure:
                        plt.savefig(resources / filename)
                        generated_plots.append(filename)
                        logger.info(f"[{metric_type}] {metric} done")
                    else:
                        logger.warning(f"[{metric_type}] {metric} NOT done:")
                        logger.warning(f"Did not save plot for metric: {metric} due to inapplicability")
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
            html_content += "<p><strong>Please note:</strong> Micro-Averages are not given for Precision, Recall and F1-Score as they correspond to (Micro) Accuracy for the multi-class case.</p>"
            df_global = pd.DataFrame.from_dict(global_metric_results, orient='index').stack().to_frame(name='Value')
            html_content += df_global.to_html(escape=False)
                
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
