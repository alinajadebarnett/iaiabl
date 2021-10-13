from matplotlib.pyplot import imsave, imread
import numpy as np
from skimage.transform import resize
import os

img_dir = "/usr/xtmp/IAIABL/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/pruned_prototypes_epoch50_k6_pt3/img/epoch-50/"
paths = []
size = 5
sizeh = 3
sizew = 5
save_dir = img_dir + "model_results_proto_visualization/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(15):
    paths.append(img_dir + "prototype-img-original_with_self_act"+ str(i) + ".png")

tosave = np.zeros((sizeh * 250, sizew * 250, 4))
index = 0
index_ = 1
for path in paths:
    try:
        arr = imread(path)
        # print("size = ", arr.shape)
    except:
        arr = np.ones((224,224,4))
    # print(arr.shape)
    # print(path)
    arr = np.pad(arr, ((13, 13), (13, 13), (0, 0)), constant_values=0)
    tosave[(index // sizew) * 250:(index // sizew) * 250 + 250, (index % sizew) * 250:(index % sizew) * 250 + 250] = arr
    index += 1
    if index == sizeh * sizew:
        imsave(save_dir+ "/"+str(index_), tosave, cmap="gray")
        tosave = np.zeros((sizeh * 250, sizew * 250))
        index_ += 1
        index = 0
        # print("Saved!")
imsave(save_dir + "/last", tosave)
print(f'Saved to {save_dir}.')

