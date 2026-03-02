# Set Base Image
FROM python:3.11.13-bookworm

# Set WORKDIR
WORKDIR /app

# COPY and install requirements
COPY build/inference/requirements_remote.txt /app/build/inference/requirements_remote.txt
RUN pip install --no-cache-dir -r build/inference/requirements_remote.txt

COPY build/inference/requirements_local.txt /app/build/inference/requirements_local.txt
RUN pip install --no-cache-dir -r build/inference/requirements_local.txt

# COPY files
# main script
COPY src/scripts/run_ensemble_inference.py /app/src/scripts/run_ensemble_inference.py

# training utils
COPY src/training_utils/lit_models.py /app/src/training_utils/lit_models.py

# shared utils
COPY src/shared_utils/custom_logging.py /app/src/shared_utils/custom_logging.py

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Define entrypoint
CMD ["python", "-O", "src/scripts/run_ensemble_inference.py", "data/BENv2.lmdb/", "data/metadata.parquet", "data/metadata_for_patches_with_snow_cloud_or_shadow.parquet", "data/checkpoints/", "results/"]
