#!/bin/bash

NUM_GPUS=8

#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=$NUM_GPUS   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=$NUM_GPUS
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "Experiment: Sentiment Control"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/slurm-%j.out

# Enable wandb in offline mode
export WANDB_MODE=offline

wandb enabled

export DATASETS_PATH="/home/rmorain2/bbc/datasets/"

accelerate launch \
    --config_file=/home/rmorain2/bbc/multi_gpu.yaml \
    --num_processes $NUM_GPUS \
    /home/rmorain2/bbc/bbc/experiment_sentiment.py