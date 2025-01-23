#!/bin/bash
#SBATCH -J modernbert
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -p h100
#SBATCH --gres=gpu:1             
#SBATCH --time 1:00:00

cd ~/ModernBERT_original
source venv/bin/activate
srun python main.py yamls/main/modernbert-base.yaml

