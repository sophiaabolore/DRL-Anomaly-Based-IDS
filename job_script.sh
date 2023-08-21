#!/bin/bash

#SBATCH --job-name=DRL_for_IDS_experiments
#SBATCH --partition=t4v2,rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# prepare your environment here
module load /usr/local/bin/python3.9

#  run the python scripts concurrently
python main.py
python QRDQN.py
python IQN.py
python C51.py

# Wait for all background jobs to complete before proceeding
wait

cp -r ./results
