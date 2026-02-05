# Set Base Image
FROM python:3.11.13-bookworm

# Set WORKDIR
WORKDIR /app

# COPY and install requirements
COPY build/train_lightning/requirements.txt /app/build/train_lightning/requirements.txt
RUN pip install --no-cache-dir -r build/train_lightning/requirements.txt

# COPY files
# main script
COPY src/scripts/train_lightning.py /app/src/scripts/train_lightning.py

# training utils
COPY src/training_utils/lit_models.py /app/src/training_utils/lit_models.py
COPY src/training_utils/train_lightning_configuration.py /app/src/training_utils/train_lightning_configuration.py
COPY src/training_utils/run_tracker.py /app/src/training_utils/run_tracker.py

# shared utils
COPY src/shared_utils/custom_logging.py /app/src/shared_utils/custom_logging.py

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Define entrypoint
CMD ["python", "-O", "src/scipts/train_lightning.py", "data/BENv2.lmdb/", "data/metadata.parquet", "data/metadata_for_patches_with_snow_cloud_or_shadow.parquet", "data/class_distribution.parquet", "models/"]
