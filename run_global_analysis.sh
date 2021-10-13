#!/bin/bash

source /home/virtual_envs/ml/bin/activate

nvidia-smi

echo "This cannot be run without the original dataset."

srun -u python3 global_analysis.py -modeldir='/usr/xtmp/mammo/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/pruned_prototypes_epoch50_k6_pt3/' \
                                   -model='50_4_0prune0.9533.pth' \
                                   -push_dir='/usr/xtmp/mammo/Lo1136i_with_fa/train_plus_val/' \
                                   -test_dir='/usr/xtmp/mammo/Lo1136i_with_fa/test/'

echo "The above is for pruned IAIA-BL"

srun -u python3 global_analysis.py -modeldir='/usr/xtmp/mammo/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/' \
                                   -model='50_4push0.9546.pth' \
                                   -push_dir='/usr/xtmp/mammo/Lo1136i_with_fa/train_plus_val/' \
                                   -test_dir='/usr/xtmp/mammo/Lo1136i_with_fa/test/'

echo "The above is for UNpruned IAIA-BL"

srun -u python3 global_analysis.py -modeldir='/usr/xtmp/mammo/saved_models/vgg16/0125_topkk=1_fa=0.0_random=4/' \
                                   -model='50_5push0.9209.pth' \
                                   -push_dir='/usr/xtmp/mammo/Lo1136i_with_fa/train_plus_val/' \
                                   -test_dir='/usr/xtmp/mammo/Lo1136i_with_fa/test/'

echo "The above is for original protopnet (Baseline 1)"