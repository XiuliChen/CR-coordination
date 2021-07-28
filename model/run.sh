#!/bin/bash

#SBATCH --job-name=reaching
#SBATCH --output=outs/task_number_%A_%a.out
#SBATCH --array=0-100
#SBATCH --time=05:00:00
#SBATCH --mem=3GB

module load anaconda
#pip install --user stable_baselines3    

n=$SLURM_ARRAY_TASK_ID
srun python run.py --n=${n}
