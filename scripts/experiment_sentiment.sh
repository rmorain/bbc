#!/bin/bash

#SBATCH --time=00:10:59   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH -J "Experiment: Sentiment Control"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/slurm-%j.out

wandb offline

export DATASETS_PATH="/home/rmorain2/bbc/datasets/"

accelerate launch \
    --config_file=/home/rmorain2/bbc/multi_gpu.yaml \
    --num_processes 2 \
    /home/rmorain2/bbc/bbc/experiment_sentiment.py --debug
