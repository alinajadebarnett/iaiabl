#!/bin/bash

source /home/virtual_envs/ml/bin/activate

nvidia-smi

echo "To get the output from paper, you must have access to the entire test set."

srun -u python3 graphing.py