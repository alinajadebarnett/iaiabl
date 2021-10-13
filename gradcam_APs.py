### Adapted from https://github.com/stefannc/GradCAM-Pytorch/blob/07fd6ece5010f7c1c9fbcc8155a60023819111d7/example.ipynb retrieved Mar 3 2021 #####

## cell 1: imports
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import make_grid, save_image

from gradcam_utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp

import torchvision.transforms as transforms
from vanilla_vgg import Vanilla_VGG
from dataHelper import DatasetFolder, DatasetFolder_WithReplacement
from skimage.transform import resize
import our_vgg
from collections import defaultdict
import argparse

## argparsing
parser = argparse.ArgumentParser()
parser.add_argument("-save_loc", type=str)
args = parser.parse_args()

## get our mammo img
test_dir = '/usr/xtmp/IAIABL/Lo1136i/test/'
test_dataset = DatasetFolder(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=False)
sample_img, target, patient_id = next(iter(test_loader)) 
print(patient_id)
normed_torch_img = sample_img.cuda()
torch_img = normed_torch_img


## cell 4: load model
# vgg = models.vgg16(pretrained=True)
model_path = '/usr/xtmp/IAIABL/saved_models/vanilla/0125_vanilla_3margin_vgg16_latent512_baseline3_random=4/0.9384582045743842_at_epoch_136'
vgg_us = torch.load(model_path)
vgg_us.eval(), vgg_us.cuda();

state_dict = vgg_us.state_dict()

for key in list(state_dict.keys()):
    state_dict[key.replace('features.features.', 'features.')] = state_dict.pop(key)

vgg_l = our_vgg.vgg16()
vgg_l.load_state_dict(state_dict)
vgg_l.eval(), vgg_l.cuda();

# Ref: https://stackoverflow.com/q/54846905/7521428
# print("### OUR MODEL ###")
# l = [module for module in vgg_us.modules() if type(module) != nn.Sequential]
# print(l)

print("### USUAL VGG MODEL ###")
vgg = models.vgg16(pretrained=True)
vgg.eval(), vgg.cuda();
# l = [module for module in vgg.modules() if type(module) != nn.Sequential]
# print(l)

cam_dict = dict()

vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='features_29', input_size=(224, 224))
vgg_gradcam = GradCAM(vgg_model_dict, True)
vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]

vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='features_6', input_size=(224, 224))
vgg_gradcam = GradCAM(vgg_model_dict, True)
vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
cam_dict['vgg_layer6'] = [vgg_gradcam, vgg_gradcampp]

vgg_us_model_dict = dict(type='vgg_us', arch=vgg_us, layer_name='features_29', input_size=(224, 224))
vgg_us_gradcam = GradCAM(vgg_us_model_dict, True)
vgg_us_gradcampp = GradCAMpp(vgg_us_model_dict, True)
cam_dict['vgg_us'] = [vgg_us_gradcam, vgg_us_gradcampp]

vgg_l_model_dict = dict(type='vgg', arch=vgg_l, layer_name='features_29', input_size=(224, 224))
vgg_l_gradcam = GradCAM(vgg_l_model_dict, True)
vgg_l_gradcampp = GradCAMpp(vgg_l_model_dict, True)
cam_dict['vgg_l'] = [vgg_l_gradcam, vgg_l_gradcampp]

## cell 5: make image grid
images = []
for gradcam, gradcam_pp in cam_dict.values():
    mask, _ = gradcam(normed_torch_img)
    # print("Min of mask is: ", torch.min(mask))
    # print("Max of mask is: ", torch.max(mask))
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img)
    # print("Max of mask_pp is: ", torch.max(mask_pp))
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
    
    images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
    
image_grid = make_grid(torch.cat(images, 0), nrow=5)

## cell 6: save image grid
output_path = args.save_loc

os.makedirs(output_path[:-8])

save_image(image_grid, output_path)

## from AP generator
def activation_precision(dataloader, # can be train, test, train_finer, test_finer
                        model,
                        gradcam,
                        num_classes=3,
                        preprocess_input_function=None,
                        log=print,
                        debug_mode=True,
                        per_class=False):

    #assert dataloader loads with fourth channel
    #assert dataloader batch size of 1

    precisions = []

    per_class_hp = defaultdict(list)

    for idx, (search_batch_input, search_y, patient_id) in enumerate(dataloader):
        if debug_mode:
            print('batch {}'.format(idx))
        if preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input[:, :3, : , :])
        else:
            search_batch = search_batch_input

        search_batch = search_batch_input[:, :3, : , :]
        fine_anno = 1 - search_batch_input[:, 3:, : , :]

        if debug_mode:
            print("search_batch:", search_batch.shape)
            print("fine_anno:", fine_anno.shape)
            print("search_y:", search_y.detach().cpu().numpy()[0])
            print("search_y.shape, sy[0]:", search_y.shape, search_y[0])
            print("fine_anno[0][0][0][0]: ", fine_anno[0][0][0][0])
            print("fine_anno[0][0][122][122]: ", fine_anno[0][0][122][122])


        with torch.no_grad():
            search_batch = search_batch.cuda()
            fine_anno = fine_anno.cuda()

        proto_acts, _ = gradcam(search_batch, class_idx=search_y.detach().cpu().numpy()[0])

        if debug_mode:
            print("proto_acts:", proto_acts.shape)

        hps = fine_anno * proto_acts

        if debug_mode:
            print("hps:", hps.shape)

        fine_anno_ = np.copy(fine_anno.detach().cpu().numpy())

        percentile = 95

        activation_map_ = proto_acts.cpu()
        threshold = np.percentile(activation_map_, percentile)
        mask = np.ones(activation_map_.shape)
        mask[activation_map_ < threshold] = 0

        if idx==0 and debug_mode:
            print("act_map:", activation_map_.shape)
            print("mask:", mask.shape)
            print("fine_anno_:", fine_anno_.shape)
        denom = np.sum(mask)
        num = np.sum(mask * fine_anno_[0][0])

        if idx==0 and debug_mode:
            print(f"act. prec. for first image is: {num/denom}")
        precisions.append(num/denom)
        per_class_hp[search_y.detach().cpu().numpy()[0]].append(num/denom)


    if per_class:
        per_class_hp_list = []
        for k, v in per_class_hp.items():
            per_class_hp_list.append((k, np.average(np.asarray(v))))
        per_class_hp_list.sort(key=lambda x: x[0])
        return per_class_hp_list
    else:
        return np.average(np.asarray(precisions))


## call the AP func for a single check

test_dir = '/usr/xtmp/IAIABL/Lo1136i_finer/by_margin/test/'
test_dataset = DatasetFolder(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=False)

num_classes = len(test_dataset.classes)

print("fine-scale activation precision for gradCAM is: ", 
        activation_precision(dataloader=test_loader, # can be train, test, train_finer, test_finer
                            model=vgg_l,
                            gradcam=vgg_l_gradcam,
                            num_classes=num_classes,
                            preprocess_input_function=None,
                            log=print,
                            debug_mode=False,
                            per_class=False)
        )

print("fine-scale activation precision for gradCAM++ is: ", 
        activation_precision(dataloader=test_loader, # can be train, test, train_finer, test_finer
                            model=vgg_l,
                            gradcam=vgg_l_gradcampp,
                            num_classes=num_classes,
                            preprocess_input_function=None,
                            log=print,
                            debug_mode=False,
                            per_class=False)
        )

test_dir = '/usr/xtmp/IAIABL/Lo1136i_with_fa/test/'
test_dataset = DatasetFolder(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=False)

print("lesion-scale activation precision for gradCAM is: ", 
        activation_precision(dataloader=test_loader, # can be train, test, train_finer, test_finer
                            model=vgg_l,
                            gradcam=vgg_l_gradcam,
                            num_classes=num_classes,
                            preprocess_input_function=None,
                            log=print,
                            debug_mode=False,
                            per_class=False)
        )

print("lesion-scale activation precision for gradCAM++ is: ", 
        activation_precision(dataloader=test_loader, # can be train, test, train_finer, test_finer
                            model=vgg_l,
                            gradcam=vgg_l_gradcampp,
                            num_classes=num_classes,
                            preprocess_input_function=None,
                            log=print,
                            debug_mode=False,
                            per_class=False)
        )


## bootstrapped AP function calls

f_test_dir = '/usr/xtmp/IAIABL/Lo1136i_finer/by_margin/test/'
l_test_dir = '/usr/xtmp/IAIABL/Lo1136i_with_fa/test/'

for test_dir in [f_test_dir, l_test_dir]:
    print(f'for data in {test_dir}')
    test_dataset = DatasetFolder_WithReplacement(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
        )

    test_batch_size = len(test_dataset.samples) # we decided to use a sample size equal to the size of the test set.
    CI = 0.95 # confidence interval
    num_iterations = 10 #DEAR REVIEWERS: In the implementation presented in the paper this value was 5000, but to make the demo faster I reduced this value.
    aps = [0]*num_iterations # doing this instead of a list append marginally improves computational efficiency
    aps_pp = [0]*num_iterations

    for i in range(num_iterations):
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=True,
                num_workers=4, pin_memory=False)
        aps[i] = activation_precision(dataloader=test_loader, # can be train, test, train_finer, test_finer
                                    model=vgg_l,
                                    gradcam=vgg_l_gradcam,
                                    num_classes=num_classes,
                                    preprocess_input_function=None,
                                    log=print,
                                    debug_mode=False,
                                    per_class=False)
        aps_pp[i] = activation_precision(dataloader=test_loader, # can be train, test, train_finer, test_finer
                                        model=vgg_l,
                                        gradcam=vgg_l_gradcampp,
                                        num_classes=num_classes,
                                        preprocess_input_function=None,
                                        log=print,
                                        debug_mode=False,
                                        per_class=False)

    vois = zip([aps, aps_pp],\
                ['GradCAM AP', 'GradCAM++ AP'])
    lower, upper = 100 * ( (1.0 - CI)/2. ), 100 *  ( 1.0 - ((1.0 - CI)/2.) )
    for valueofinterest, valueofinterest_str in vois:
        voi_mean = np.mean(np.asarray(valueofinterest))
        voi_std = np.std(np.asarray(valueofinterest))
        voi_lower, voi_upper = np.percentile(valueofinterest, [lower, upper])
        print(f"Final mean {valueofinterest_str} is {voi_mean}, std {voi_std} {CI*100}% confidence iterval accuracy is {voi_lower} to {voi_upper} with {num_iterations} iterations.")