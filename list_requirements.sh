#!/bin/bash

source /home/virtual_envs/ml/bin/activate

srun -u python -m pip freeze
