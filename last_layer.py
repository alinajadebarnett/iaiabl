import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from dataHelper import DatasetFolder
import re
import numpy as np
import os
import copy
from skimage.transform import resize
from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

def show_last_layer_connections(ppnet):
    print(ppnet.num_prototypes, ppnet.num_classes)
    last_layer_connections = np.zeros((ppnet.num_prototypes, ppnet.num_classes))
    last_layer_connections = ppnet.last_layer.weight
    return last_layer_connections

def show_last_layer_connections_T(ppnet):
    print(ppnet.num_prototypes, ppnet.num_classes)
    last_layer_connections = np.zeros((ppnet.num_prototypes, ppnet.num_classes))
    last_layer_connections = ppnet.last_layer.weight
    last_layer_connections_T = torch.transpose(last_layer_connections, 0, 1)
    return last_layer_connections_T