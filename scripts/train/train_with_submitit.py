# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import submitit
import argparse
import socket
import torch
from composer import Trainer

from omegaconf import OmegaConf as om


def set_up_dist_env(port):
    # 1. RANK
    job_env = submitit.JobEnvironment()
    global_rank = job_env.global_rank

    # 2. LOCAL_RANK
    local_rank = job_env.local_rank

    # 3. LOCAL_WORLD_SIZE
    ngpus_per_node = torch.cuda.device_count()

    # 4. WORLD_SIZE
    world_size = int(os.getenv("SLURM_NNODES")) * ngpus_per_node

    # 5. NODE_RANK
    node_rank = int(os.getenv("SLURM_NODEID"))

    # 6. MASTER_ADDR
    cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0]

    # Set All the Necessary Environment Variables!
    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(ngpus_per_node)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["MASTER_ADDR"] = host_name
    os.environ["MASTER_PORT"] = str(port)
    os.environ["PYTHONUNBUFFERED"] = "1"

def main(args) -> Trainer:

    set_up_dist_env(args.port)

    with open(args.config_file) as f:
        cfg = om.load(f)

    if args.model_type == "bert":
        from train_bert import main as train
    elif args.model_type == "causal":
        from train import main as train
    
    train(cfg)
    

def submit_job(args):
    executor = submitit.AutoExecutor(folder='./slurm')
    executor.update_parameters(
        slurm_job_name=args.name,
        slurm_time =args.time,
        nodes=args.num_nodes,
        gpus_per_node=args.num_gpus_per_node,
        tasks_per_node=args.num_gpus_per_node,
        cpus_per_task=args.workers,
        mem_gb=args.num_gpus_per_node*args.mem_per_gpu,
        slurm_constraint=args.gpu_type,
        slurm_setup=["module purge", "module load cuda/11.8"]
    )
    job = executor.submit(main, args)

    print('Slurm Job ID: ', job.job_id)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Arguments for submitit"
    )

    parser.add_argument(
        "--name", type=str, required=True, help="Slurm job name."
    )
    parser.add_argument(
        "--model_type", type=str, choices=["bert", "causal"], required=True, help="Type of the model."
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to config file."
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        required=True,
        help="Number of nodes to request.",
    )
    parser.add_argument(
        "--num_gpus_per_node",
        type=int,
        required=True,
        help="Number of GPUs to request per node.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=True,        
        help="Number of dataloader workers to request.",
    )
    parser.add_argument(
        "--mem_per_gpu",
        type=int,
        default=64,        
        help="RAM memory to request per GPU in GB.",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        required=True,
        choices=["v100", "a100"],       
        help="Type of GPU to request.",
    )
    parser.add_argument(
        "--time",
        type=int,
        required=True,        
        help="Time in minutes.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12345,
        help="Port for disatributed training.",
    )

    args = parser.parse_args()

    submit_job(args)

