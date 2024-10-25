#!/bin/bash

#SBATCH --time=5:10:59   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=128G   # memory per CPU core
#SBATCH -J "Evaluate"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/sentiment_control/slurm-%j-eval.out

# Pass in policy model as a command line argument with gpt2 as default
POLICY_MODEL="${1:-gpt2}"

wandb offline

export DATASETS_PATH="$PWD/datasets/"

# Simple debug
accelerate launch \
    --config_file=$PWD/multi_gpu.yaml \
    --num_processes 8 \
    $PWD/bbc/sentiment_evaluate.py \
    --policy_model $POLICY_MODEL \
    --base_models meta-llama/Meta-Llama-3.1-8B \
    --batch_size 8 \
    --mini_batch_size 8 \
    --num_generations 25 \
    --description "Evaluate $POLICY_MODEL" \