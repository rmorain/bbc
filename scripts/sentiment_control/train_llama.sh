#!/bin/bash

#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=128G   # memory per CPU core
#SBATCH -J "Llama as base model"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/sentiment_control/llama/slurm-%j-train.out

wandb offline

export DATASETS_PATH="$PWD/datasets/"

accelerate launch \
    --config_file=$PWD/multi_gpu.yaml \
    --num_processes 8 \
    $PWD/bbc/sentiment_train.py \
    --num_epochs 1 \
    --policy_model gpt2 \
    --base_models meta-llama/Meta-Llama-3.1-8B \
    --dataset sst2_processed \
    --description "Llama as base model" \

# Read the model name from the file
MODEL_NAME=$(cat $PWD/checkpoints/$SLURM_JOB_ID/model_name.txt)

sbatch --job-name=eval_${MODEL_NAME} scripts/sentiment_control/evaluate_llama.sh $MODEL_NAME