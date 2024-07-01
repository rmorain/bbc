#!/bin/bash

#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "Test train"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/slurm-%j.out
#
# Enable wandb in offline mode
export WANDB_MODE=offline

wandb enabled

export DATASETS_PATH="/home/rmorain2/bbc/datasets/"

# Set a specific directory for wandb files (adjust the path as needed)
export WANDB_DIR="/home/rmorain2/bbc/wandb"

accelerate launch --config_file=/home/rmorain2/bbc/multi_gpu.yaml /home/rmorain2/bbc/bbc/train.py

# After the job completes, print a reminder message
echo "Job completed. Remember to sync wandb data from a node with internet access."
echo "Run 'wandb sync $WANDB_DIR' to upload the results."