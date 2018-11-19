#!/bin/bash
#SBATCH -A support-gpu
#SBATCH -p pascal
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --exclusive
#SBATCH -t 12:00:00 

source runenv

if [[ $SLURM_NNODES -gt 1 ]]
then
  srun bash tf-wrapper.sh python $@
else
  bash tf-wrapper.sh python $@
fi
