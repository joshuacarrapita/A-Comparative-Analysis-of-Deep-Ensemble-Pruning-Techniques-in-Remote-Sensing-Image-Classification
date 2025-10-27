#!/bin/bash

# --- Job Configuration ---
#SBATCH -J train_lightning_ensemble  # Job name for the array
#SBATCH --array=0-99                 # Create 100 tasks, indexed 0-99
## %10                                # Optional: Limit to 10 concurrent tasks

# --- Resource Request (Lowest Common Denominator) ---
# Request resources that are available on ALL target partitions
# to ensure the job can be scheduled on any of them.
# Partitions: gpu, gpu_short, scioi_gpu, scioi_a100nv
# CPUs: min(12, 8) = 8
# Memory: min(500G, 360G) = 360G
#SBATCH --partition=gpu,gpu_short,scioi_gpu,scioi_a100nv
#SBATCH --nodes=1                    # Request 1 node per task
#SBATCH --ntasks-per-node=1          # Request 1 task (our python script)
#SBATCH --cpus-per-task=8            # Request 8 CPUs (min of your configs)
#SBATCH --gpus-per-task=1            # Request 1 GPU (generic)
#SBATCH --mem=360G                   # Request 360G (min of your configs)
#SBATCH --time=04:00:00              # Max walltime for EACH task

# --- Logging (to RAM, then move to SCRATCH) ---
# %A is the Array Job ID, %a is the Array Task ID
# This creates a unique log file for each task in RAM.
#SBATCH --output=/run/shm/ensemble_%A_%a.out
#SBATCH --error=/run/shm/ensemble_%A_%a.err

# --- Mail Notifications ---
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joshuacar.rapita@campus.tu-berlin.de

# ===================================================================
# --- Job Script Start ---
# ===================================================================

# --- 1. Setup Environment ---
echo "Loading environment modules..."
module load singularity/4.0.2
module load nvidia/cuda/12.2

# Set thread environment variables (1 or $SLURM_CPUS_PER_TASK-1 or -2 to leave these threads for the main process and other processes) if one worker should be allowed to take up all the CPU resources momentarily.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create the final destination log directory on scratch
export LOG_DIR="/scratch/joshuacar/out"
mkdir -p $LOG_DIR

echo "Starting job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID"

# --- 2. Run Training Script ---
# This script should be aware of SLURM_ARRAY_TASK_ID to set seeds
srun /beegfs/scratch/joshuacar/programs/train_lightning/train_lightning_test.sh

echo "Training script finished."

# --- 3. Move Logs from RAM to Scratch ---
# Use the SLURM variables to find the unique log files
echo "Moving log files to $LOG_DIR..."
srun mv /run/shm/ensemble_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out $LOG_DIR/
srun mv /run/shm/ensemble_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err $LOG_DIR/

echo "Job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID finished."
