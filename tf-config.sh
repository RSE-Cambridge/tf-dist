#!/bin/bash

scontrol show hostname | python tf-config.py $SLURM_PROCID
