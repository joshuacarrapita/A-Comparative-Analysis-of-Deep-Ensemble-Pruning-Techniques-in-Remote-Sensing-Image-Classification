#!/bin/bash

## -----------------------------------------------------------------
## ---- SBATCH COMMAND INFO  ----
##
## ---- OUTPUT FILE DEFs ----
##SBATCH -o /run/shm/train_lightning-test.%j.%N.out  # for RAM output 
##SBATCH -o /run/shm/%j/out/train_lightning-test.%j.%N.out  # Did not work (folder creation with %j failed)
##SBATCH -o /scratch/joshuacar/out/train_lightning-test.%j.%N.out  # for `scratch` output 
##
## ---- JOB NAME ----
##SBATCH -J train_lightning-test
##
## ---- MAX. JOB RUNTIME ----
##SBATCH --time=04:00:00 	# Assumed runtime (0:20:00 * ntasks)
##
## ---- SELECTED PARTITION ----
##SBATCH --partition=standard  	# 14d
##SBATCH --partition=TestAndBuild  	# 5h
##SBATCH --partition=scioi_node  	# 14d
##SBATCH --partition=gpu  	# 7d
##SBATCH --partition=gpu_short 	# 1d
##SBATCH --partition=scioi_gpu  	# 7d
##SBATCH --partition=scioi_a100nv  	# 7d
##
## ---- MAIL SETTINGS FOR JOB STATUS ----
##SBATCH --mail-type=ALL
##SBATCH --mail-user=joshua.carrapita@campus.tu-berlin.de
##
## ---- AMOUNT PROCESSES ----
##SBATCH --ntasks=1 		
##
## ---- AMOUNT CPUS / TASK ----
##SBATCH --cpus-per-task=8	# SET THIS TO the amount of cores for one socket to let someone else have the cores on the other socket 
##
## ---- AMOUNT RAM / NODE ----
##SBATCH --mem=360G             # 1500M * ntasks
##
## ---- REQUESTED GPU ----
##SBATCH --gpus=a100:1		# A100
##SBATCH --gres=gpu:a100:1	# A100 using gres (generic resource)
##SBATCH --gres=gpu:v100s:1	# V100 "
##SBATCH --gres=gpu:tesla:2	# P100 x 2 "
##
## -----------------------------------------------------------------

## -----------------------------------------------------------------
## ---- SHARED SBATCH COMMANDs ----
#SBATCH -o /run/shm/train_lightning-test.%j.%N.out   
#SBATCH -J train_lightning-test 		
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joshua.carrapita@campus.tu-berlin.de
##
## ---- UNCOMMENT FOR GPU & GPU_SHORT ----
#SBATCH --partition=gpu_short
##SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G              
#SBATCH --gres=gpu:a100:1	
##SBATCH --gres=gpu:a100:2	
##SBATCH --gres=gpu:tesla:2	
##
## ---- UNCOMMENT FOR SCIOI_GPU ----
##SBATCH --partition=scioi_gpu
##SBATCH --cpus-per-task=8	
##SBATCH --mem=360G             # Max. 370G
##SBATCH --gres=gpu:a100:1	
##SBATCH --gres=gpu:v100s:1	
##SBATCH --gres=gpu:v100s:2	
##
## ---- UNCOMMENT FOR STANDARD & TestAndBuild ----
##SBATCH --partition=standard
##SBATCH --cpus-per-task=10	
##SBATCH --mem=250G             

# benötigte SW / Bibliotheken laden
module load singularity/4.0.2
module load nvidia/cuda/12.2

export OMP_NUM_THREADS=1 # or set to $SLURM_CPUS_PER_TASK (-> -1 or -2 to leave these threads for the main process and other processes) if one worker should be allowed to take up all the CPU resources momentarily, same for the next one
export MKL_NUM_THREADS=1 
# export CUDA_LAUNCH_BLOCKING=1 # set this if you use the profiler because asynchronous computation on CUDA devices can mess up the calculated runtimes for operations

srun /beegfs/scratch/joshuacar/programs/train_lightning/train_lightning_test.sh
#echo "Moving out logs to scratch ..."
#srun mv /run/shm/${SLURM_JOB_ID}/out/ /scratch/joshuacar/out/
echo "Moving out logs to scratch ..."
srun mv /run/shm/train_lightning-test.${SLURM_JOB_ID}.${SLURM_NODELIST}.out /scratch/joshuacar/out/
