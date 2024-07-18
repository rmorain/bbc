#!/bin/bash

#SBATCH --time=0:10:59   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH -J "Experiment: Sentiment Control"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/slurm-%j.out

wandb offline

export DATASETS_PATH="/home/rmorain2/bbc/datasets/"

accelerate launch \
    --config_file=/home/rmorain2/bbc/multi_gpu.yaml \
    --num_processes 1 \
    /home/rmorain2/bbc/bbc/experiment_sentiment.py \
    --num_epochs 1 \
    --policy_model gpt2 \
    --base_models gpt2 gpt2-medium \
    --description Testing two base models at once \
    --debug \
