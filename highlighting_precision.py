import torch
import numpy as np

import heapq

import matplotlib.pyplot as plt
import os
import copy
import time

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop, silent_print
from find_nearest import ImagePatch, ImagePatchInfo

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from dataHelper import DatasetFolder
import re
from skimage.transform import resize
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse
import pandas as pd
import ast
import png

from collections import defaultdict
from pathlib import Path

from matplotlib.pyplot import imsave, imread
from skimage.color import rgb2hsv, hsv2rgb
from copy import copy

def highlighting_precision(dataloader, # can be train, test, train_finer, test_finer
                            prototype_network_parallel, # pytorch network with prototype_vectors
                            ppnet,
                            load_model_dir,
                            epoch_number_str,
                            preprocess_input_function=None,
                            log=print,
                            prototype_activation_function_in_numpy=None,
                            debug_mode=True,
                            per_proto=False):

    #assert dataloader loads with fourth channel

    n_prototypes = prototype_network_parallel.module.num_prototypes

    precisions = []

    per_proto_hp = defaultdict(list)

    for idx, (search_batch_input, search_y, patient_id) in enumerate(dataloader):
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
            print("search_y.shape, sy[0]:", search_y.shape, search_y[0])
            print("fine_anno[0][0][0][0]: ", fine_anno[0][0][0][0])
            print("fine_anno[0][0][122][122]: ", fine_anno[0][0][122][122])


        with torch.no_grad():
            search_batch = search_batch.cuda()
            fine_anno = fine_anno.cuda()
            protoL_input_torch, proto_dist_torch = \
                prototype_network_parallel.module.push_forward(search_batch)

        proto_acts = ppnet.distance_2_similarity(proto_dist_torch)

        proto_acts = torch.nn.Upsample(size=(search_batch.shape[2], search_batch.shape[3]), mode='bilinear', align_corners=False)(proto_acts)

        if debug_mode:
            print("proto_acts:", proto_acts.shape)

        # confirm prototype class identity
        load_img_dir = os.path.join(load_model_dir, 'img')

        prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
        prototype_img_identity = prototype_info[:, -1]

        if debug_mode:
            log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
            log('Their class identities are: ' + str(prototype_img_identity))

        hps = fine_anno * proto_acts

        if debug_mode:
            print("hps:", hps.shape)

        proto_acts_ = np.copy(proto_acts.detach().cpu().numpy())
        hps_ = np.copy(hps.detach().cpu().numpy())
        fine_anno_ = np.copy(fine_anno.detach().cpu().numpy())

        percentile = 95

        for img_idx, activation_map in enumerate(proto_acts_):
            # for every test img
            for j in range(n_prototypes):
                # for each proto
                if prototype_img_identity[j] == search_y[img_idx]:
                    # if proto class matches img class

                    activation_map_ = activation_map[j]
                    threshold = np.percentile(activation_map_, percentile)
                    mask = np.ones(activation_map_.shape)
                    mask[activation_map_ < threshold] = 0

                    if img_idx==0 and debug_mode:
                        print(search_y[img_idx])
                        print("act_map:", activation_map_.shape)
                        print("mask:", mask.shape)
                        print("fine_anno_:", fine_anno_.shape)
                    denom = np.sum(mask)
                    num = np.sum(mask * fine_anno_[img_idx][0])

                    if debug_mode and False:
                        print(f"hp is: {num/denom}")
                    precisions.append(num/denom)
                    per_proto_hp[j].append(num/denom)
    if per_proto:
        per_proto_hp_list = []
        for k, v in per_proto_hp.items():
            per_proto_hp_list.append((k, np.average(np.asarray(v))))
        per_proto_hp_list.sort(key=lambda x: x[0])
        return per_proto_hp_list
    else:
        return np.average(np.asarray(precisions))

def overlayed_img(original_img, upsampled_activation_pattern):
    # show the image overlayed with prototype activation map
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    overlayed_img = 0.5 * original_img + 0.3 * heatmap
    return overlayed_img

def highlighting_precision_visualization(dataloader, # can be train, test, train_finer, test_finer
                                        prototype_network_parallel, # pytorch network with prototype_vectors
                                        ppnet,
                                        load_model_dir,
                                        epoch_number_str,
                                        preprocess_input_function=None,
                                        log=print,
                                        prototype_activation_function_in_numpy=None,
                                        debug_mode=True,
                                        per_proto=False):

    #assert dataloader loads with fourth channel

    n_prototypes = prototype_network_parallel.module.num_prototypes

    precisions = []

    per_proto_hp = defaultdict(list)

    for idx, (search_batch_input, search_y, patient_id) in enumerate(dataloader):
        print('batch {}'.format(idx))
        if preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input[:, :3, : , :])
        else:
            search_batch = search_batch_input

        search_batch = search_batch_input[:, :3, : , :]
        orig_img = search_batch_input[:, 0, : , :] 
        orig_img = np.copy(orig_img.detach().cpu().numpy())
        fine_anno = 1 - search_batch_input[:, 3:, : , :]

        if debug_mode:
            print("search_batch:", search_batch.shape)
            print("fine_anno:", fine_anno.shape)
            print("search_y.shape, sy[0]:", search_y.shape, search_y[0])
            print("fine_anno[0][0][0][0]: ", fine_anno[0][0][0][0])
            print("fine_anno[0][0][122][122]: ", fine_anno[0][0][122][122])


        with torch.no_grad():
            search_batch = search_batch.cuda()
            fine_anno = fine_anno.cuda()
            protoL_input_torch, proto_dist_torch = \
                prototype_network_parallel.module.push_forward(search_batch)

        proto_acts = ppnet.distance_2_similarity(proto_dist_torch)

        proto_acts = torch.nn.Upsample(size=(search_batch.shape[2], search_batch.shape[3]), mode='bilinear', align_corners=False)(proto_acts)

        if debug_mode:
            print("proto_acts:", proto_acts.shape)

        # confirm prototype class identity
        load_img_dir = os.path.join(load_model_dir, 'img')

        prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
        prototype_img_identity = prototype_info[:, -1]

        if debug_mode:
            log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
            log('Their class identities are: ' + str(prototype_img_identity))

        hps = fine_anno * proto_acts

        if debug_mode:
            print("hps:", hps.shape)

        proto_acts_ = np.copy(proto_acts.detach().cpu().numpy())
        hps_ = np.copy(hps.detach().cpu().numpy())
        fine_anno_ = np.copy(fine_anno.detach().cpu().numpy())

        percentile = 95.0
        percentile_ = percentile / 100

        save_vis_dir = load_model_dir
        save_vis_dir = os.path.join(save_vis_dir, 'visualizations_of_hp/')
        save_vis_dir = os.path.join(save_vis_dir, f'prec={percentile_:.2f}/')
        print("save dir:", save_vis_dir)
        
        if not os.path.exists(save_vis_dir):
            os.makedirs(save_vis_dir)

        for img_idx, activation_map in enumerate(proto_acts_):
            # for every test img
            for j in range(n_prototypes):
                # for each proto
                if prototype_img_identity[j] == search_y[img_idx]:
                    # if proto class matches img class

                    activation_map_ = activation_map[j]
                    threshold = np.percentile(activation_map_, percentile)
                    mask = np.ones(activation_map_.shape)
                    mask[activation_map_ < threshold] = 0

                    if img_idx==0 and debug_mode:
                        print(search_y[img_idx])
                        print("act_map:", activation_map_.shape)
                        print("mask:", type(mask), mask.shape)
                        print("fine_anno_:", fine_anno_.shape)
                        print("orig_img:", type(orig_img[img_idx]), orig_img[img_idx].shape)
                    denom = np.sum(mask)
                    num = np.sum(mask * fine_anno_[img_idx][0])

                    prec = num/denom

                    if img_idx==0 and debug_mode:
                        print(f"hp is: {prec}")

                    precisions.append(prec)
                    per_proto_hp[j].append(prec)

                    # getting nice looking images with hsv
                    saturation = 0.8
                    mask_hue = 1
                    anno_hue = 0.6
                    both_hue = 0.8

                    orig = np.transpose(np.stack([orig_img[img_idx], orig_img[img_idx], orig_img[img_idx]]), (1,2,0))
                    imsave(save_vis_dir + f'{idx}-{img_idx}orig{prec:.2f}.png', orig)

                    img_with_act = overlayed_img(orig, activation_map_)
                    if img_idx==0 and debug_mode:
                        print("img_with_act:", img_with_act.shape)
                    imsave(save_vis_dir + f'{idx}-{img_idx}orig_act{prec:.2f}.png', img_with_act)

                    mask_img = rgb2hsv(copy(orig))
                    mask_img[:, :, 1] = saturation * mask
                    mask_img[:, :, 0] = mask_hue * mask
                    mask_img = hsv2rgb(mask_img)
                    imsave(save_vis_dir + f'{idx}-{img_idx}act-mask{prec:.2f}.png', mask_img)

                    fa_img = rgb2hsv(copy(orig))
                    fa_img[:, :, 1] = saturation * fine_anno_[img_idx][0]
                    fa_img[:, :, 0] = anno_hue * fine_anno_[img_idx][0]
                    fa_img = hsv2rgb(fa_img)
                    imsave(save_vis_dir + f'{idx}-{img_idx}fa{prec:.2f}.png', fa_img)

                    both_img = rgb2hsv(copy(orig))
                    both_img[:, :, 1] = saturation * np.maximum(fine_anno_[img_idx][0], mask)
                    both_img[:, :, 0] = mask_hue * (mask - np.minimum(fine_anno_[img_idx][0], mask)) \
                                        + anno_hue * (fine_anno_[img_idx][0] - np.minimum(fine_anno_[img_idx][0], mask)) \
                                        + both_hue * (np.minimum(fine_anno_[img_idx][0], mask))
                    both_img = hsv2rgb(both_img)
                    imsave(save_vis_dir + f'{idx}-{img_idx}both{prec:.2f}.png', both_img)
    if per_proto:
        per_proto_hp_list = []
        for k, v in per_proto_hp.items():
            per_proto_hp_list.append((k, np.average(np.asarray(v))))
        per_proto_hp_list.sort(key=lambda x: x[0])
        return per_proto_hp_list
    else:
        return np.average(np.asarray(precisions))

def hp(test_dir, load_model_path, per_proto=False, verbose=False):

    pathlib_path_load_model_path = Path(load_model_path)
    load_model_name = pathlib_path_load_model_path.parts[-1]
    load_model_dir = pathlib_path_load_model_path.parent

    regex = re.compile(r'(?P<epoch>[0-9]+)(_|nopush).*')
    mo = regex.fullmatch(load_model_name)
    assert mo is not None
    epoch_number_str = mo.group('epoch')

    if verbose:
        print('load model from ' + load_model_path)
        print('test set directory: ' + test_dir)
    
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    class_specific = True

    normalize = transforms.Normalize(mean=mean,
                                 std=std)
    test_batch_size = 100

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
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    if verbose:
        print('test set size: {0}'.format(len(test_loader.dataset)))

    if verbose:
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=print)
        print('accu', str(accu))

    # get hp
    return (highlighting_precision(dataloader=test_loader, # can be train, test, train_finer, test_finer
                                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                                ppnet=ppnet,
                                load_model_dir=load_model_dir,
                                epoch_number_str=epoch_number_str,
                                preprocess_input_function=None,
                                log=silent_print,
                                prototype_activation_function_in_numpy=None,
                                debug_mode=False,
                                per_proto=per_proto))

def get_highlighting_precision(test_dir, load_model_dir, load_model_name, per_proto=False, viz=False):

    check_test_accu = True

    model_base_architecture = load_model_dir.split('/')[-3]
    experiment_run = load_model_dir.split('/')[-2]

    save_analysis_path = load_model_dir + 'hp/' + load_model_name + '/'
    os.makedirs(save_analysis_path, exist_ok=True)

    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'highlighting_precision.log'))

    load_model_path = os.path.join(load_model_dir, load_model_name)
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)

    log('load model from ' + load_model_path)
    log('test set directory: ' + test_dir)
    log('save analysis to: ' + save_analysis_path)

    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    class_specific = True

    # load the test data and check test accuracy
    if check_test_accu:
        test_batch_size = 100

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
            test_dataset, batch_size=test_batch_size, shuffle=True,
            num_workers=4, pin_memory=False)
        log('test set size: {0}'.format(len(test_loader.dataset)))

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=print)
        log(str(accu))

    # get hp
    if viz:
        return (highlighting_precision_visualization(dataloader=test_loader, # can be train, test, train_finer, test_finer
                                                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                                                    ppnet=ppnet,
                                                    load_model_dir=load_model_dir,
                                                    epoch_number_str=epoch_number_str,
                                                    preprocess_input_function=None,
                                                    log=print,
                                                    prototype_activation_function_in_numpy=None,
                                                    debug_mode=True,
                                                    per_proto=per_proto))
    else:
        return (highlighting_precision(dataloader=test_loader, # can be train, test, train_finer, test_finer
                                        prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                                        ppnet=ppnet,
                                        load_model_dir=load_model_dir,
                                        epoch_number_str=epoch_number_str,
                                        preprocess_input_function=None,
                                        log=print,
                                        prototype_activation_function_in_numpy=None,
                                        debug_mode=True,
                                        per_proto=per_proto))

def main():

    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_directory', nargs=1, type=str, default='0')
    parser.add_argument('-model_dir', nargs=1, type=str, default='0')
    parser.add_argument('-model_name', nargs=1, type=str, default='0')
    args = parser.parse_args()

    test_dir =  args.test_directory[0]

    load_model_dir = args.model_dir[0]
    load_model_name = args.model_name[0]

    # get hp
    print(get_highlighting_precision(test_dir, load_model_dir, load_model_name, per_proto=False, viz=False))
              
if __name__ == "__main__":
    main()
    print("Ended.")