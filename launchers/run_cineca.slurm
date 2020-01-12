#!/bin/bash

#SBATCH --job-name=IJCAI_TEST
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --account=IscrC_TBD
#SBATCH --time=8:00:00
#SBATCH --partition=gll_usr_gpuprod
#SBATCH --gres=gpu:kepler:1

python3 -W ignore main.py

