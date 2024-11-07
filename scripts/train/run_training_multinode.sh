#!/bin/bash -l

cd /ibex/ai/project/c2254/zehua/llm-foundry/scripts/train
source ~/.bashrc
conda activate llm

python train_with_submitit.py --name mosaicbert \
                              --model_type bert \
                              --config_file ./yamls/pretrain/zehua-bert.yaml \
                              --num_nodes 2 \
                              --num_gpus_per_node 4 \
                              --workers 6 \
                              --mem_per_gpu 64 \
                              --gpu_type a100 \
                              --time 1440 \
                              --port 1235
