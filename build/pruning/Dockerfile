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
COPY src/scripts/run_ensemble_pruning.py src/scripts/run_ensemble_pruning.py

# pruning submodules
COPY src/pruning_submodules/ src/pruning_submodules/

# define path of installed modules
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# define entrypoint
ENTRYPOINT ["python", "-O", "src/scripts/run_ensemble_pruning.py", "--results", "data/inference/results.pt", "--targets", "data/inference/target_labels.pt", "--outdir", "data/pruning/out/"]
