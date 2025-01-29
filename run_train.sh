#!/bin/bash
#SBATCH -J modernbert
#SBATCH -c 64
#SBATCH --mem=500G
#SBATCH -p h100
#SBATCH --gres=gpu:8             
#SBATCH --time 1-00:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user anton.ehrmanntraut@uni-wuerzburg.de

cd ~/ModernBERT_original
source venv/bin/activate

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export JOBNAME=modernbert_base_${SLURM_JOB_ID}

srun /bin/bash -c "export NODE_RANK=\$SLURM_NODEID; composer -n 8 --stderr output_${JOBNAME}_{rank}.out --stdout output_${JOBNAME}_{rank}.out main.py yamls/main/modernbert-base.yaml run_name=$JOBNAME 2>&1 | awk -v node_rank=\$(cat /etc/hostname) '{ print node_rank, strftime(\"%c: \"), \$0; fflush(); }' | tee -a output_${JOBNAME}_0.out"

