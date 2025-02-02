#!/bin/bash

#SBATCH --job-name="modernbert_distributed"
#SBATCH --partition h100
#SBATCH --tasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mail-type ALL
#SBATCH --mail-user anton.ehrmanntraut@uni-wuerzburg.de

export MASTER_PORT=29400
export WORLD_SIZE=16
#export LOCAL_RANK=
export LOCAL_WORLD_SIZE=8
#export NODE_RANK=
#export MASTER_ADDR=
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_PROTO=simple
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

cd /home/ane53vq/ModernBERT_original/
source venv/bin/activate

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export JOBNAME=modernbert_1b_${SLURM_JOB_ID}

srun /bin/bash -c "export NODE_RANK=\$SLURM_NODEID; composer -n 8 main.py yamls/main/modernbert-1b.yaml run_name=$JOBNAME 2>&1 | awk -v node_rank=\$(cat /etc/hostname) '{ print node_rank, strftime(\"%c: \"), \$0; fflush(); }'"
