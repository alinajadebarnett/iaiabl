# IAIA-BL

This code implements IAIA-BL from the manuscript "A Case-based 
Interpretable Deep Learning Model for Classification of Mass Lesions 
in Digital Mammography" published in Nature Machine Intelligence, Dec 2021, 
by Alina Jade Barnett, Fides Regina Schwartz, Chaofan Tao, Chaofan Chen, 
Yinhao Ren, Joseph Y. Lo, and Cynthia Rudin.

This code package was developed by the authors at Duke University and 
University of Maine, and licensed as described in LICENSE (for more 
information regarding the use and the distribution of this code package).

## Prerequisites
Any operating system on which you can run GPU-accelerated 
PyTorch. Python 3.6.9. For packages see requirements.txt.
### Recommended hardware
2 NVIDIA Tesla P-100 GPUs or 2 NVIDIA Tesla V-100 GPUs

## Installation instructions
1. Git clone the repository to /usr/xtmp/IAIABL/. 
3. Set up your environment using Python 3.6.9 and requirements.txt. 
   (Optional) Set up your environment using requirements.txt so that "source
   /home/virtual_envs/ml/bin/activate" activates your environment. You can 
   set up the environment differently if you choose, but all .sh scripts 
   included will attempt to activate the environment at 
   /home/virtual_envs/ml/bin/activate.
Typical install time: Less than 10 minutes.

## Train the model
1. In train.sh, the appropriate file locations should be set for train_dir, 
test_dir, push_dir and finer_dir:
   1. train_dir is the directory containing the augmented training set
   2. test_dir is the directory containing the test set
   3. push_dir is the directory containing the original (unaugmented) training 
   set, onto which prototypes can be projected
   4. finer_dir is the directory containing the augmented set of training 
   examples with fine-scale annotations

2. Run train.sh

## Reproducing figures
No data is provided with this code repository. The following scripts are 
included to demonstrate how figures and results were created for the 
paper. The following scripts require data to be provided. Type "source 
scriptname.sh" into the command line to run.

1. see_explanations.sh

Expected output from see_explanations.sh are figures from the 
manuscript that begin with "An automatically generated explanation 
of mass margin classification." The paths to the output images will 
appear in the relative file location "./visualizations_of_expl/".

2. see_prototype_grid.sh

Expected output from see_prototype_grid.sh will be a grid of prototypes 
for a given model. The file location where the output image can be 
found will be printed onto the command line.

3. run_gradCAM.sh

Expected output from run_gradCAM.sh will show the activation precision of the
sample data. It will also print a visualization in 
/usr/xtmp/IAIABL/gradCAM_imgs/view.png. The columns from left to right are 
"Original Image," "GradCAM heatmap," "GradCAM++ heatmap," "GradCAM heatmap 
overlayed on the original image," and "GradCAM++ heatmap overlayed on the 
original image." The rows are "Last layer, using a network trained on natural
images," "6th layer, using a network trained on natural images," "Blank," and 
"Last layer, using a network trained to identify the mass margin."

4. The mal_for_reviewers.ipynb Jupyter notebook is also included.

Expected output from mal_for_reviewers.ipynb is in the cells of the notebook.

Expected run time for these four demo files: 10 minutes.

## Other functions
The following scripts require the more of the (private) dataset in order to 
run correctly, but are included to aid in reproducibility:
1. dataaugment.sh - for offline data augmentation
2. plot_graph.sh - plots a variety of graphs
3. run_global_analysis.sh - provides a global analysis of the model
4. train_vanilla_malignancy.sh - for training the baseline models

## Expected Data Location
Scripts are set up to expect data as numpy arrays in 
/usr/xtmp/IAIABL/Lo1136i/test/Circumscribed/ where Circumscribed is the 
mass margin label. The first channel of the numpy array should be image 
data and the second (optional) channel should be the fine annotation 
label.
