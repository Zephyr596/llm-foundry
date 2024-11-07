#!/bin/bash -l
#SBATCH --job-name=mpt350m-arabictext2022-contextlength512
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=256G
#SBATCH --constraint=[a100]

module load cuda/11.8

conda activate llm

cd /ibex/ai/project/c2254/zehua/llm-foundry/scripts/train

# BERT
composer train_bert.py ./yamls/pretrain/zehua-bert.yaml
