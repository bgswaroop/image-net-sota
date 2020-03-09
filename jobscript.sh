#!/bin/bash

#SBATCH --job-name=AlexNetVanilla
#SBATCH --time=51:00:00
#SBATCH --mem=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# echo starting_jobscript
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

# echo activating environment
source venv/bin/activate
pip freeze

# echo running_the_flow
python run_flow.py
# echo completed_the_job