#!/bin/bash

#SBATCH --time=5:10:59   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "Sentiment control"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/slurm-%j.out

# Pass in policy model as a command line argument with gpt2 as default
POLICY_MODEL="${1:-gpt2}"

wandb offline

export DATASETS_PATH="$PWD/datasets/"
# export SLURM_JOB_ID="test"

# Simple debug
accelerate launch \
    --config_file=$PWD/multi_gpu.yaml \
    --num_processes 1 \
    $PWD/bbc/sentiment_evaluate.py \
    --policy_model $POLICY_MODEL \
    --base_models gpt2 \
    --description "Debug single model" \