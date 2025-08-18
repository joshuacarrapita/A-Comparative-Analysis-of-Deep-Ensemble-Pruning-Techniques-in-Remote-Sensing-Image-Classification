#!/bin/bash

#SBATCH -o /scratch/joshuacar/out/train-test.%j.%N.out   # Output-File
#SBATCH -J train-test 		# Job Name
#SBATCH --ntasks=1 		# Anzahl Prozesse (CPU-Cores)
#SBATCH --cpus-per-task=10	# Anzahl CPU-Cores pro Prozess
#SBATCH --mem=10G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=0:30:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=standard

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joshua.carrapita@campus.tu-berlin.de

# benötigte SW / Bibliotheken laden
module load singularity/4.0.2
module load nvidia/cuda/12.2

echo "Running train.py on a single CPU core"
srun singularity run --cwd /app --bind /scratch/joshuacar/mockdata/small/BENv2:/app/data/ --bind /scratch/joshuacar/models/:/app/models/ train_0.1.0.sif 
