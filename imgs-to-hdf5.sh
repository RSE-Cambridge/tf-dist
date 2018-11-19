#!/bin/bash
set -euo pipefail

for i in ecoset imagenet
do
  export SBATCH_ACCOUNT=support-cpu
  export SBATCH_PARTITION=skylake
  export SBATCH_EXCLUSIVE=
  export SBATCH_TIME=12:00:00
  export SBATCH_JOB_NAME=imgs-to-hdf5_$i.out
  sbatch -o imgs-to-hdf5_$i.out run.sh imgs-to-hdf5.py ~/rds/rds-hpc-support/rse/full_$i{.h5,}
done
