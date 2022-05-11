#!/bin/bash

#SBATCH --job-name=Symbolic_DNN-Tuner
#SBATCH --nodes=1
#SBATCH --account=IscrC_ML4COVID
#SBATCH --time=24:00:00
#SBATCH --partition=m100_usr_prod
#SBATCH --gres=gpu:1

python main.py