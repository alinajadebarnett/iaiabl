#!/bin/bash

source /home/virtual_envs/ml/bin/activate

nvidia-smi

echo "Begun generating explanations."

MODELFOLDER=/usr/xtmp/IAIABL/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/pruned_prototypes_epoch50_k6_pt3
MODELNAME=50_4_0prune0.9533.pth

for FILENAME in DP_AJOU_197104_1.npy DP_AKKN_7728_1.npy 
do
srun -u python local_analysis.py -test_img_name "$FILENAME" \
                                -test_img_dir '/usr/xtmp/IAIABL/Lo1136i/test/Circumscribed/' \
                                -test_img_label 0 \
                                -test_model_dir "$MODELFOLDER/" \
                                -test_model_name "$MODELNAME" &>/dev/null

srun -u python3 local_analysis_vis.py -local_analysis_directory "$MODELFOLDER/$FILENAME/"
done

for FILENAME in DP_AKAY_89028_1.npy DP_AKVP_18401_1.npy DP_ALFQ_28102_1.npy
do
srun -u python local_analysis.py -test_img_name "$FILENAME" \
                                -test_img_dir '/usr/xtmp/IAIABL/Lo1136i/test/Spiculated/' \
                                -test_img_label 2 \
                                -test_model_dir "$MODELFOLDER/" \
                                -test_model_name "$MODELNAME" &>/dev/null

srun -u python3 local_analysis_vis.py -local_analysis_directory "$MODELFOLDER/$FILENAME/"
done

echo "End."