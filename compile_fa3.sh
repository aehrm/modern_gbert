#!/bin/bash
#SBATCH -J cuda_compiler
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -p standard
#SBATCH --gres=gpu:1             
#SBATCH --tmp=16G 
#SBATCH --time 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user anton.ehrmanntraut@uni-wuerzburg.de

cd ~/ModernBERT_original
source venv/bin/activate
cd flash-attention/hopper/
export MAX_JOBS=4
srun python setup.py build -j 4

