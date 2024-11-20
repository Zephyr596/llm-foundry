#!/bin/bash -l
#SBATCH --job-name=zehua-mosaic-1117
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=256G
#SBATCH --constraint=[a100]
#SBATCH --mail-type=ALL

module load cuda/11.8

source ~/.bashrc
conda activate llm

cd /ibex/ai/project/c2254/zehua/llm-foundry/scripts/train

# BERT
composer train_bert.py ./yamls/pretrain/zehua-bert.yaml
