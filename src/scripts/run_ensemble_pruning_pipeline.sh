#!/bin/bash

# define variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LMDB_FILE="data/BENv2.lmdb"
METADATA_PARQUET="data/metadata.parquet"
METADATA_SNOW_CLOUD_PARQUET="data/metadata_for_patches_with_snow_cloud_or_shadow.parquet"
MODELS_DIR="models/"

EXPORTS_DIR_INF_FULL_ENSEMBLE="results/inference/${TIMESTAMP}/full_ensemble"
EXPORTS_DIR_INF_SUBENSEMBLES="results/inference/${TIMESTAMP}"
EXPORTS_DIR_PRUNE="results/pruning/${TIMESTAMP}"
EXPORTS_DIR_EVAL="results/evaluation/${TIMESTAMP}"

FULL_ENSEMBLE_INF_RESULTS="${EXPORTS_DIR_INF_FULL_ENSEMBLE}/ensemble_inference_results.pt"
FULL_ENSEMBLE_INF_TARGET_LABELS="${EXPORTS_DIR_INF_FULL_ENSEMBLE}/ensemble_inference_target_labels.pt"

# create directories
mkdir -p $EXPORTS_DIR_INF_FULL_ENSEMBLE
mkdir -p $EXPORTS_DIR_PRUNE
mkdir -p $EXPORTS_DIR_EVAL

# run pipeline
# TODO: change run_ensemble_evaluation.py to run over the results and targets 
# for all subensembles in the $EXPORTS_DIR_INF_SUBENSEMBLES
python -O src/scripts/run_ensemble_inference.py $LMDB_FILE $METADATA_PARQUET $METADATA_SNOW_CLOUD_PARQUET $MODELS_DIR $EXPORTS_DIR_INF_FULL_ENSEMBLE $INF_ARGS && \
python -O src/scripts/run_ensemble_pruning.py --results $FULL_ENSEMBLE_INF_RESULTS --targets $FULL_ENSEMBLE_INF_TARGET_LABELS --outdir $EXPORTS_DIR_PRUNE $PRUNE_ARGS && \
bash src/scripts/run_subensemble_inference.sh --pruning-results $EXPORTS_DIR_PRUNE --outdir $EXPORTS_DIR_INF_SUBENSEMBLES $INF_ARGS && \
python -O src/scripts/run_ensemble_evaluation.py --inferences $EXPORTS_DIR_INF_SUBENSEMBLES --outdir $EXPORTS_DIR_EVAL $EVAL_ARGS
