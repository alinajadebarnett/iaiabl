#!/bin/bash

source /home/virtual_envs/ml/bin/activate

nvidia-smi

srun -u python vanilla_vgg.py -model="vgg16"  \
                              -train_dir="/usr/xtmp/IAIABL/Lo1136i/bymal/train/" \
                              -test_dir="/usr/xtmp/IAIABL/Lo1136i/bymal/test/"\
                              -name="0202_vanilla_2mal_vgg16_latent512_random=12"\
                              -lr="1e-5" \
                              -wd="1e-1" \
                              -num_classes="2"