#!/bin/bash
#SBATCH -A support-gpu
#SBATCH -p pascal
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --exclusive
#SBATCH -t 12:00:00 

module purge
module load slurm
module load cuda/9.0 cudnn/7.3_cuda-9.0
source env/bin/activate
export LD_LIBRARY_PATH=$HOME/tf-dist/nccl_2.2.13-1+cuda9.0_x86_64/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO

if [[ $SLURM_NNODES -gt 1 ]]
then
  srun bash tf-wrapper.sh python $@
else
  bash tf-wrapper.sh python $@
fi
