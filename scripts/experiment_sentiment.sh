#!/bin/bash

#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=128G   # memory per CPU core
#SBATCH -J "Sentiment control"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/slurm-%j.out

wandb offline

export DATASETS_PATH="$PWD/datasets/"

# accelerate launch \
#     --config_file=$PWD/multi_gpu.yaml \
#     --num_processes 8 \
#     $PWD/bbc/experiment_sentiment.py \
#     --num_epochs 1 \
#     --policy_model gpt2 \
#     --base_models gpt2-large \
#     --dataset sst2_processed \
#     --description "Replicating single model control. Continuation score only. " \

# accelerate launch \
#     --config_file=$PWD/multi_gpu.yaml \
#     --num_processes 8 \
#     $PWD/bbc/experiment_sentiment.py \
#     --num_epochs 1 \
#     --policy_model gpt2 \
#     --base_models meta-llama/Meta-Llama-3.1-8B \
#     --dataset sst2_processed \
#     --description "Llama 3.1-8B as base model." \

# accelerate launch \
#     --config_file=$PWD/multi_gpu.yaml \
#     --num_processes 8 \
#     $PWD/bbc/experiment_sentiment.py \
#     --num_epochs 5 \
#     --policy_model gpt2 \
#     --base_models gpt2 gpt2-medium \
#     --dataset imdb_sst2_processed \
#     --description "Controlling two models" \

# accelerate launch \
#     --config_file=$PWD/multi_gpu.yaml \
#     --num_processes 1 \
#     $PWD/bbc/experiment_sentiment.py \
#     --num_epochs 1 \
#     --policy_model gpt2 \
#     --base_models gpt2 gpt2-medium \
#     --dataset imdb_sst2_processed \
#     --description "Debug" \
#     --debug \

# accelerate launch \
#     --config_file=$PWD/multi_gpu.yaml \
#     --num_processes 1 \
#     $PWD/bbc/experiment_sentiment.py \
#     --num_epochs 1 \
#     --policy_model gpt2 \
#     --base_models gpt2 \
#     --dataset sst2_processed \
#     --description "Debug single model" \
#     --debug \

# accelerate launch \
#     --config_file=$PWD/multi_gpu.yaml \
#     --num_processes 1 \
#     $PWD/bbc/experiment_sentiment.py \
#     --num_epochs 1 \
#     --policy_model gpt2 \
#     --base_models gpt2-large \
#     --dataset imdb_sst2_tokenized \
#     --description "Replicating single model control. Continuation score only. " \

# python \
#     $PWD/bbc/experiment_sentiment.py \
#     --num_epochs 3 \
#     --policy_model gpt2 \
#     --base_models gpt2-large \
#     --dataset imdb_sst2_tokenized \
#     --description "Replicating single model control. Continuation score only. " \

accelerate launch \
    --config_file=$PWD/multi_gpu.yaml \
    --num_processes 1 \
    $PWD/bbc/experiment_sentiment.py \
    --num_epochs 1 \
    --policy_model gpt2 \
    --base_models gpt2-large \
    --dataset imdb_sst2_tokenized \
    --lr 1.41e-3 \
    --description "Replicating single model control. Continuation score only. " \