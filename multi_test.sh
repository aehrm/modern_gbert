#!/bin/bash

#SBATCH --output slurm_output_%N
#SBATCH --error slurm_output_%N
#SBATCH --job-name="distributed_test"
#SBATCH --partition h100
#SBATCH --tasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=32G

export MASTER_PORT=29400
export WORLD_SIZE=4
#export LOCAL_RANK=
export LOCAL_WORLD_SIZE=2
#export NODE_RANK=
#export MASTER_ADDR=
export PYTHONUNBUFFERED=1

cd /home/ane53vq/ModernBERT_original/
source venv/bin/activate

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

#srun /bin/bash -c "nvidia-smi && echo \$MASTER_ADDR \$SLURM_NODEID && cat /etc/hostname"

srun /bin/bash -c "export NODE_RANK=\$SLURM_NODEID; cat /etc/hostname; composer main.py yamls/main/modernbert-base.yaml "
