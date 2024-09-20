#!/bin/bash

#SBATCH --time=0:10:59   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "Debug train"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/sentiment_control/debug/slurm-%j-train.out

wandb offline

export DATASETS_PATH="$PWD/datasets/"
# export SLURM_JOB_ID="test"

# Simple debug
accelerate launch \
    --config_file=$PWD/multi_gpu.yaml \
    --num_processes 1 \
    $PWD/bbc/sentiment_train.py \
    --num_epochs 1 \
    --policy_model gpt2 \
    --base_models gpt2 \
    --dataset hello_world_2048 \
    --description "Debug single model" \
    --debug \