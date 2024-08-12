#!/bin/bash


#SBATCH --time=10:09:59   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "Evaluate"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/slurm-%j.out

wandb offline
export DATASETS_PATH="$PWD/datasets/"

# accelerate launch \
    #  --config_file=$PWD/bbc/multi_gpu.yaml \
#     --num_processes 8 \
#     /home/rmorain2/bbc/bbc/evaluate.py \
#     --policy_model /home/rmorain2/bbc/saved_models/gpt2_f55v2cw1 \

accelerate launch \
    --config_file=$PWD/multi_gpu.yaml \
    --num_processes 8 \
    $PWD/bbc/evaluate.py \
    --policy_model gpt2 \