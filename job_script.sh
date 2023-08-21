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

# Use srun to run the python scripts concurrently
srun --ntasks=1 python main.py &
srun --ntasks=1 python IQN.py &
srun --ntasks=1 python QRDQN.py &
srun --ntasks=1 python C51.py &

# Wait for all background jobs to complete before proceeding
wait

cp -r ./results
