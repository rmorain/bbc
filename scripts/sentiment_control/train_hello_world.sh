#!/bin/bash

#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=128G   # memory per CPU core
#SBATCH -J "Replicating different water"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/sentiment_control/hello-world-%j-train.out

wandb disabled

export DATASETS_PATH="$PWD/datasets/"

accelerate launch \
    --config_file=$PWD/multi_gpu.yaml \
    --num_processes 1 \
    $PWD/bbc/sentiment_train.py \
    --num_epochs 3 \
    --batch_size 1 \
    --mini_batch_size 1 \
    --policy_model gpt2 \
    --base_models gpt2-large \
    --dataset hello_world_2048 \
    --description "Train on a single example" \

# Read the model name from the file
MODEL_NAME=$(cat $PWD/checkpoints/$SLURM_JOB_ID/model_name.txt)

# sbatch --job-name=eval_${MODEL_NAME} scripts/sentiment_control/evaluate.sh $MODEL_NAME