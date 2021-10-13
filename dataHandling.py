from __future__ import division
import numpy as np
import os
import pandas as pd
import argparse
import sys
import random
import png
from matplotlib.pyplot import imsave, imread
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
matplotlib.use("Agg")
import torchvision.datasets as datasets
from skimage.transform import resize
import ast
import pickle
import csv
import pydicom as dcm
import Augmentor
from tqdm import tqdm
import pathlib
from torch import randint, manual_seed
from copy import copy
from collections import defaultdict

def random_flip(input, axis, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            axis += 1
        return np.flip(input, axis=axis)
    else:
        return input

def random_crop(input, with_fa=False):
    ran = random.random()
    if ran > 0.2:
        # find a random place to be the left upper corner of the crop
        if with_fa:
            rx = int(random.random() * input.shape[1] // 10)
            ry = int(random.random() * input.shape[2] // 10)
            return input[:, rx: rx + int(input.shape[1] * 9 // 10), ry: ry + int(input.shape[2] * 9 // 10)]
        else:
            rx = int(random.random() * input.shape[0] // 10)
            ry = int(random.random() * input.shape[1] // 10)
            return input[rx: rx + int(input.shape[0] * 9 // 10), ry: ry + int(input.shape[1] * 9 // 10)]
    else:
        return input

def random_rotate_90(input, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            return np.rot90(input, axes=(1,2))
        return np.rot90(input)
    else:
        return input

def random_rotation(x, chance, with_fa=False):
    ran = random.random()
    if with_fa:
        img = Image.fromarray(x[0])
        mask = Image.fromarray(x[1])
        if ran > 1 - chance:
            # create black edges
            angle = np.random.randint(0, 90)
            img = img.rotate(angle=angle, expand=1)
            mask = mask.rotate(angle=angle, expand=1, fillcolor=1)
            return np.stack([np.asarray(img), np.asarray(mask)])
        else:
            return np.stack([np.asarray(img), np.asarray(mask)])
    img = Image.fromarray(x)
    if ran > 1 - chance:
        # create black edges
        angle = np.random.randint(0, 90)
        img = img.rotate(angle=angle, expand=1)
        return np.asarray(img)
    else:
        return np.asarray(img)

def augment_numpy_images(path, targetNumber, targetDir, skip=None, rot=True, with_fa=False):
    classes = os.listdir(path)
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    for class_ in classes:
        if not os.path.exists(targetDir + class_):
            os.makedirs(targetDir + class_)

    for class_ in classes:
        count, round = 0, 0
        while count < targetNumber:
            round += 1
            for root, dir, files in os.walk(os.path.join(path, class_)):
                for file in files:
                    if skip and skip in file:
                        continue
                    filepath = os.path.join(root, file)
                    arr = np.load(filepath)
                    print("loaded ", file)
                    print(arr.shape)
                    try:
                        arr = random_crop(arr, with_fa)
                        print(arr.shape)
                        if rot:
                            arr = random_rotation(arr, 0.9, with_fa)
                        print(arr.shape)
                        arr = random_flip(arr, 0, with_fa)
                        arr = random_flip(arr, 1, with_fa)
                        arr = random_rotate_90(arr, with_fa)
                        arr = random_rotate_90(arr, with_fa)
                        arr = random_rotate_90(arr, with_fa)
                        print(arr.shape)
                        if with_fa:
                            whites = arr.shape[2] * arr.shape[1] - np.count_nonzero(np.round(arr[0] - np.amax(arr[0]), 2))
                            black = arr.shape[2] * arr.shape[1] - np.count_nonzero(np.round(arr[0], 2))
                            if arr.shape[2] < 10 or arr.shape[1] < 10 or black >= arr.shape[2] * arr.shape[1] * 0.8 or \
                                whites >= arr.shape[2] * arr.shape[1] * 0.8:
                                print("illegal content")
                                continue

                        else:
                            whites = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr - np.amax(arr), 2))
                            black = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr, 2))

                            if arr.shape[0] < 10 or arr.shape[1] < 10 or black >= arr.shape[0] * arr.shape[1] * 0.8 or \
                                    whites >= arr.shape[0] * arr.shape[1] * 0.8:
                                print("illegal content")
                                continue

                        if count % 10 == 0:
                            if not os.path.exists("./visualizations_of_augmentation/" + class_ + "/"):
                                os.makedirs("./visualizations_of_augmentation/" + class_ + "/")
                            if with_fa:
                                imsave("./visualizations_of_augmentation/" + class_ + "/" + str(count), np.transpose(np.stack([arr[0], arr[0], arr[1]]), (1,2,0)))
                            else:
                                imsave("./visualizations_of_augmentation/" + class_ + "/" + str(count), np.transpose(np.stack([arr, arr, arr]), (1,2,0)))


                        np.save(targetDir + class_ + "/" + file[:-4] + "aug" + str(round), arr)
                        count += 1
                        print(count)
                    except:
                        print("something is wrong in try, details:", sys.exc_info()[2])
                        if not os.path.exists("./error_of_augmentation/" + class_ + "/"):
                            os.makedirs("./error_of_augmentation/" + class_ + "/")
                        np.save("./error_of_augmentation/" + class_ + "/" + str(count), arr)
                    if count > targetNumber:
                        break
    print(count)

def window_augmentation(wwidth, wcen):
    if wcen == 2047 and wwidth == 4096:
        return wwidth, wcen
    else:
        new_wcen = np.random.randint(-100, 300)
        new_wwidth = np.random.randint(-200, 300)
        wwidth += new_wwidth
        wcen += new_wcen
        return wwidth, wcen

if __name__ == "__main__":

    print("Data augmentation")
    for pos in ["Spiculated","Circumscribed", "Indistinct"]:
        augment_numpy_images(
            path="/usr/xtmp/mammo/npdata/datasetname_with_fa/train/",
            targetNumber=5000,
            targetDir="/usr/xtmp/mammo/npdata/datasetname_with_fa/train_augmented_5000/",
            rot=True,
            with_fa=True)

