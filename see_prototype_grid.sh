#!/bin/bash

source /home/virtual_envs/ml/bin/activate

nvidia-smi

srun -u python3 vis_protos.py
