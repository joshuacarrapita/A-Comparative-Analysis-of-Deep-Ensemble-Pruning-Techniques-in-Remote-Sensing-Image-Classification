# Set Base Image
FROM python:3.11.13-bookworm

# Set WORKDIR
WORKDIR /app

# COPY and install requirements
COPY build/requirements_train_lightning.txt /app/build/requirements_train_lightning.txt
RUN pip install --no-cache-dir -r build/requirements_train_lightning.txt

# COPY files
COPY src/train_lightning.py /app/src/train_lightning.py

# Define entrypoint
CMD ["python", "-O", "src/train_lightning.py", "data/BENv2.lmdb/", "data/metadata.parquet", "data/metadata_for_patches_with_snow_cloud_or_shadow.parquet", "models/"]
