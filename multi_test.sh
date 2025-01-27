#!/bin/bash

#SBATCH --output slurm_output_%N
#SBATCH --error slurm_output_%N
#SBATCH --job-name="distributed_test"
#SBATCH --partition h100
#SBATCH --tasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:2
#SBATCH -c 24
#SBATCH --mail-type ALL
#SBATCH --mail-user anton.ehrmanntraut@uni-wuerzburg.de

export MASTER_PORT=29400
export WORLD_SIZE=4
#export LOCAL_RANK=
export LOCAL_WORLD_SIZE=2
#export NODE_RANK=
#export MASTER_ADDR=
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_PROTO=simple
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

cd /home/ane53vq/ModernBERT_original/
source venv/bin/activate

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

#srun /bin/bash -c "nvidia-smi && echo \$MASTER_ADDR \$SLURM_NODEID && cat /etc/hostname"

srun /bin/bash -c "export NODE_RANK=\$SLURM_NODEID; cat /etc/hostname; composer --stdout 'logdir/log_rank{rank}' --stderr 'logdir/log_rank{rank}' main.py yamls/main/modernbert-base.yaml "
