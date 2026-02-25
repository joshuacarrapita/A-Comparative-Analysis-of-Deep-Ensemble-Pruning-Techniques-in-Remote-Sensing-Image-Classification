# set base image
FROM python:3.11.13-bookworm

# set WORKDIR
WORKDIR /app

# COPY and install requirements
COPY build/pruning/requirements_remote.txt build/pruning/requirements_remote.txt
RUN pip install --no-cache-dir -r build/pruning/requirements_remote.txt

COPY build/pruning/requirements_local.txt build/pruning/requirements_local.txt
RUN pip install --no-cache-dir -r build/pruning/requirements_local.txt

# COPY files
# main script
COPY src/scripts/run_ensemble_pruning_pipeline.sh src/scripts/run_ensemble_pruning_pipeline.sh
COPY src/scripts/run_ensemble_inference.py src/scripts/run_ensemble_inference.py
COPY src/scripts/run_subensemble_inference.sh src/scripts/run_subensemble_inference.sh
COPY src/scripts/run_ensemble_pruning.py src/scripts/run_ensemble_pruning.py
COPY src/scripts/run_ensemble_evaluation.py src/scripts/run_ensemble_evaluation.py

# pruning submodules
COPY src/pruning_submodules/ src/pruning_submodules/

# shared utils
COPY src/shared_utils/custom_logging.py src/shared_utils/custom_logging.py

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# define entrypoint
ENTRYPOINT ["bash", "src/scripts/run_ensemble_pruning_pipeline.sh"]
