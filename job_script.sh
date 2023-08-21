#!/bin/bash
SBATCH --job-name=DRL_for_IDS_experiments

SBATCH --partition=t4v2,rtx6000,a40

SBATCH --gres=gpu:1

SBATCH --qos=normal

SBATCH --cpus-per-task=1

SBATCH --mem-per-cpu=2G

SBATCH --output=slurm-%j.out

SBATCH --error=slurm-%j.err

# prepare your environment here
module load /usr/local/bin/python3.9

# put your command here
python main.py

#python DQN.py
#
#python IQN.py

python QRDQN.py

#python C51.py

cp -r ./results
