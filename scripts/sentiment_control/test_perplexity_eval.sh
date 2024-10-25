#!/bin/bash

#SBATCH --time=5:59:59   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=128G   # memory per CPU core
#SBATCH -J "test perplexity eval"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/bbc/logs/sentiment_control/slurm-%j-train.out
python bbc/test_perplexity_eval.py