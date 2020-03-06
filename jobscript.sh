#!/bin/bash

#SBATCH --job-name=AlexNetVanilla
#SBATCH --time=00:15:00
#SBATCH --mem=2000
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1

# echo starting_jobscript
TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

# echo activating environment
source venv/bin/activate
pip freeze

# echo running_the_flow
python run_flow.py
# echo completed_the_job