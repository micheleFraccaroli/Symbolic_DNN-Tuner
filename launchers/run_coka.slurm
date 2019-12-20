#!/bin/bash

#SBATCH --job-name=DeepGAN
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4

python3 -W ignore main.py