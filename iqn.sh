#!/bin/bash

#SBATCH --job-name=Main_IDS_experiment
#SBATCH --partition=t4v2,rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm-main-%j.out
#SBATCH --error=slurm-main-%j.err

# prepare your environment here
module load /usr/local/bin/python3.9

# Run main.py
python IQN.py

cp -r ./results/iqn
