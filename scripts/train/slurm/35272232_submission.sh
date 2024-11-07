#!/bin/bash

# Parameters
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=6
#SBATCH --error=/ibex/project/c2254/zehua/llm-foundry/scripts/train/slurm/%j_0_log.err
#SBATCH --gpus-per-node=4
#SBATCH --job-name=mosaicbert
#SBATCH --mem=256GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --open-mode=append
#SBATCH --output=/ibex/project/c2254/zehua/llm-foundry/scripts/train/slurm/%j_0_log.out
#SBATCH --signal=USR2@90
#SBATCH --time=1440
#SBATCH --wckey=submitit

# setup
module purge
module load cuda/11.8

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /ibex/project/c2254/zehua/llm-foundry/scripts/train/slurm/%j_%t_log.out --error /ibex/project/c2254/zehua/llm-foundry/scripts/train/slurm/%j_%t_log.err /ibex/user/caoz0a/conda-environments/llm/bin/python -u -m submitit.core._submit /ibex/project/c2254/zehua/llm-foundry/scripts/train/slurm
