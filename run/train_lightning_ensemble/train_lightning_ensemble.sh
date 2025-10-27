#!/bin/bash

# ===================================================================
# --- 1. Define Paths ---
# ===================================================================

# --- Source Paths on Scratch ---
SRC_SANDBOX="/scratch/joshuacar/programs/train_lightning/train_lightning_0.1.1_sandbox"
SRC_LMDB="/scratch/hackelle/data/BigEarthNet-V2/BENv2.lmdb/"
SRC_META="/scratch/joshuacar/data/full/BENv2/metadata.parquet"
SRC_SNOW_META="/scratch/joshuacar/data/full/BENv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet"
SRC_CLASS_COUNTS="/scratch/joshuacar/data/original/label_counts_metadata.parquet"

# --- Destination on Scratch ---
FINAL_MODEL_DIR="/scratch/joshuacar/models/"

# --- RAM Disk Paths (Unique Per Task) ---
# Use SLURM_ARRAY_TASK_ID to create a unique directory on the node's RAM disk
# This is critical to prevent multiple tasks on the same node from conflicting
RAM_DISK_DIR="/run/shm/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RAM_MODEL_DIR="${RAM_DISK_DIR}/models"

# --- Container Paths ---
# These are the paths *inside* the singularity container
# NOTE: These are defined in the Dockerfile's CMD. We only bind to them.
CONTAINER_LMDB_PATH="/app/data/BENv2.lmdb/"
CONTAINER_META_PATH="/app/data/metadata.parquet"
CONTAINER_SNOW_META_PATH="/app/data/metadata_for_patches_with_snow_cloud_or_shadow.parquet"
CONTAINER_CLASS_COUNTS_PATH="/app/data/class_distribution.parquet"
CONTAINER_MODEL_PATH="/app/models/" # This is the export root for the python script

# ===================================================================
# --- 2. Setup Target Dirs & Cleanup ---
# ===================================================================

echo "Creating directories on RAM disk: ${RAM_DISK_DIR}"
mkdir -p "${RAM_MODEL_DIR}" # Python script will create subdirs here

# --- Setup Cleanup Function ---
cleanup() {
    echo "Copying saved models from ${RAM_MODEL_DIR} to ${FINAL_MODEL_DIR}"
    # Copy contents of the models dir (which are the seed-specific folders)
    cp -r "${RAM_MODEL_DIR}/." "${FINAL_MODEL_DIR}/"
    echo "Cleaning up RAM disk directory: ${RAM_DISK_DIR}"
    rm -rf "${RAM_DISK_DIR}"
}
trap cleanup EXIT

# ===================================================================
# --- 3. Copy Data to RAM ---
# ===================================================================

echo "Copying data to RAM disk..."
cp "${SRC_SANDBOX}" "${RAM_DISK_DIR}/"
cp -r "${SRC_LMDB}" "${RAM_DISK_DIR}/BENv2.lmdb" # Copy the whole dir
cp "${SRC_META}" "${RAM_DISK_DIR}/metadata.parquet"
cp "${SRC_SNOW_META}" "${RAM_DISK_DIR}/metadata_for_patches_with_snow_cloud_or_shadow.parquet"
cp "${SRC_CLASS_COUNTS}" "${RAM_DISK_DIR}/label_counts_metadata.parquet" # Keep original name for bind
echo "Copying data complete."

# ===================================================================
# --- 4. Run Sandbox ---
# ===================================================================

# Set the SEED environment variable for Singularity
# SINGULARITYENV_ prepends "SEED" to the container's environment
export SINGULARITYENV_SEED=${SLURM_ARRAY_TASK_ID}

echo "Starting Singularity container..."
echo "Seed for container is: ${SINGULARITYENV_SEED}"

singularity run --nv --cwd /app \
    --bind "${RAM_DISK_DIR}/BENv2.lmdb/":${CONTAINER_LMDB_PATH} \
    --bind "${RAM_DISK_DIR}/metadata.parquet":${CONTAINER_META_PATH} \
    --bind "${RAM_DISK_DIR}/metadata_for_patches_with_snow_cloud_or_shadow.parquet":${CONTAINER_SNOW_META_PATH} \
    --bind "${RAM_DISK_DIR}/label_counts_metadata.parquet":${CONTAINER_CLASS_COUNTS_PATH} \
    --bind "${RAM_MODEL_DIR}/":${CONTAINER_MODEL_PATH} \
    "${RAM_DISK_DIR}/$(basename ${SRC_SANDBOX})"

# The sandbox image will now run its internal CMD, e.g.:
# python -O src/train_lightning.py data/BENv2.lmdb/ ... models/
# The python script must be modified to read the 'SEED' env var

echo "Singularity job finished."
