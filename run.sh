#!/bin/bash
#SBATCH -A support-gpu
#SBATCH -p pascal
#SBATCH --nodes 4
#SBATCH --ntasks 4
#SBATCH --exclusive
#SBATCH -t 1:00:00 

source run.env
echo "SLURM_NNODES: $SLURM_NNODES"
if [[ $SLURM_NNODES -gt 1 ]]
then
  srun bash tf-wrapper.sh python $@
else
  bash tf-wrapper.sh python $@
fi
