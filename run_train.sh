#!/bin/bash
#SBATCH -J modernbert
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -p h100
#SBATCH --gres=gpu:1             
#SBATCH --time 1:00:00

cd ~/ModernBERT_original
source venv/bin/activate

export JOBNAME=modernbert_base_${SLURM_JOB_ID}

srun /bin/bash -c "export NODE_RANK=\$SLURM_NODEID; composer main.py yamls/main/modernbert-base.yaml run_name=$JOBNAME 2>&1 | awk -v node_rank=\$(cat /etc/hostname) '{ print node_rank, strftime(\"%c: \"), \$0; fflush(); }'"
