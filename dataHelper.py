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

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images


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


class DatasetFolder(datasets.DatasetFolder):
    def __init__(self, root, loader, augmentation=False, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, target_size=(224, 224)):

        super(DatasetFolder, self).__init__(root, loader, ("npy",),
                                            transform=transform,
                                            target_transform=target_transform, )
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.augment = augmentation
        self.target_size = target_size
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        patient_id = path.split("/")[-1][:-4]
        sample = self.loader(path)
        if len(sample.shape) == 3:
            if self.target_size:
                sample = np.stack([resize(sample[0], self.target_size), resize(sample[1], self.target_size)])
            temp = [sample[0], sample[0], sample[0], sample[1]]
        else:
            if self.target_size:
                sample = resize(sample, self.target_size)
            if self.augment:
                sample = random_rotation(sample, 0.7)
            temp = [sample, sample, sample]
        n = np.stack(temp)
        if self.transform is not None:
            sample = self.transform(n)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print("after transform", sample.shape)
        return sample.float(), target, patient_id


class DatasetFolder_WithReplacement(datasets.DatasetFolder):
    def __init__(self, root, loader, augmentation=False, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, target_size=(224, 224)):

        super(DatasetFolder_WithReplacement, self).__init__(root, loader, ("npy",),
                                                transform=transform,
                                                target_transform=target_transform, )
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.augment = augmentation
        self.target_size = target_size
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        index = randint(0, len(self.samples), (1,))[0] #pull with replacement
        path, target = self.samples[index]
        patient_id = path.split("/")[-1][:-4]
        sample = self.loader(path)
        if len(sample.shape) == 3:
            if self.target_size:
                sample = np.stack([resize(sample[0], self.target_size), resize(sample[1], self.target_size)])

            temp = [sample[0], sample[0], sample[0], sample[1]]
        else:
            if self.target_size:
                sample = resize(sample, self.target_size)
            if self.augment:
                sample = random_rotation(sample, 0.7)
            temp = [sample, sample, sample]
        n = np.stack(temp)
        if self.transform is not None:
            sample = self.transform(n)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print("after transform", sample.shape)
        return sample.float(), target, patient_id