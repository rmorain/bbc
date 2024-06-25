#!/bin/bash

#SBATCH --time=0:59:59   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH -J "Test train"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/slurm-%j.out
#
wandb enabled
accelerate launch --config_file=/home/rmorain2/bbc/multi_gpu.yaml --num_processes 2 /home/rmorain2/bbc/bbc/train.py