#!/bin/bash
#SBATCH -J modernbert_mds
#SBATCH -c 64
#SBATCH --mem=32G
#SBATCH -p large_cpu
#SBATCH --time 12:00:00

cd ~/ModernBERT_original
source venv/bin/activate
srun python src/my_convert_dataset.py

