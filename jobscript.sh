#!/bin/bash

#SBATCH --job-name=AlexNetVanilla
#SBATCH --time=06:00:00
#SBATCH --mem=2000

# echo starting_jobscript
module load Python/3.6.4-foss-2019a
module load cuDNN/7.4.2.24-CUDA-10.0.130

# echo activating environment
source venv/bin/activate
pip freeze

# echo running_the_flow
python3 run_flow.py
# echo completed_the_job