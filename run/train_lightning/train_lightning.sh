#!/bin/bash

# --- Setup Target DIR ---
RAM_DISK_DIR="/run/shm/${SLURM_JOB_ID}"
echo "Creating directories ..."
mkdir -p "${RAM_DISK_DIR}/models/"

# --- Setup Cleanup Function ---
cleanup() {
	echo "Copying saved models ..."
	cp -r "${RAM_DISK_DIR}/models/." /scratch/joshuacar/models/
	echo "Cleaning up RAM disk directory: ${RAM_DISK_DIR}"
	rm -rf "${RAM_DISK_DIR}"
}
trap cleanup EXIT

# --- Copy Data to RAM ---
echo "Copying data to RAM disk directory: ${RAM_DISK_DIR} ..."
cp -r /scratch/joshuacar/programs/train_lightning/train_lightning_0.1.1_sandbox "${RAM_DISK_DIR}"
cp -r /scratch/hackelle/data/BigEarthNet-V2/BENv2.lmdb/ "${RAM_DISK_DIR}/"
cp /scratch/joshuacar/data/full/BENv2/metadata.parquet "${RAM_DISK_DIR}/" 
cp /scratch/joshuacar/data/full/BENv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet "${RAM_DISK_DIR}/" 
cp /scratch/joshuacar/data/original/label_counts_metadata.parquet "${RAM_DISK_DIR}/"
echo "Copying data complete."

# --- Run Sandbox ---
singularity run --nv --cwd /app \
	--bind "${RAM_DISK_DIR}/metadata.parquet":/app/data/metadata.parquet \
	--bind "${RAM_DISK_DIR}/metadata_for_patches_with_snow_cloud_or_shadow.parquet":/app/data/metadata_for_patches_with_snow_cloud_or_shadow.parquet \
	--bind "${RAM_DISK_DIR}/label_counts_metadata.parquet":/app/data/class_distribution.parquet \
	--bind "${RAM_DISK_DIR}/BENv2.lmdb/":/app/data/BENv2.lmdb/ \
	--bind "${RAM_DISK_DIR}/models/":/app/models/ \
	"${RAM_DISK_DIR}/train_lightning_0.1.1_sandbox"

echo "Singularity job finished."
