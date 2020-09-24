# -------------------------------------------------
# The library is migrated from the repository:
# https://github.com/steverab/failing-loudly
# -------------------------------------------------
import os
import numpy as np
import scipy.io
from math import ceil
from enum import Enum
from keras.datasets import mnist, cifar10, cifar100, boston_housing, fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from torchvision import datasets, transforms
import torch
from kstest.log import getLogger

logger = getLogger(__name__)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:{}'.format(1))
else:
    DEVICE = torch.device('cpu')


# -------------------------------------------------
# Path UTILS
# -------------------------------------------------

class PathManager:
    def __init__(self, dateset):
        self.base_folder = f"results"
        self.dataset_folder = "dataset"
        self.model_folder = "saved_models"
        self.dataset = dateset
        os.makedirs(self.base_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)


# -------------------------------------------------
# DATA UTILS
# -------------------------------------------------
def set_random_seed():
    import random
    import tensorflow
    seed = 1
    # tensorflow.set_random_seed(seed)
    np.random.seed(1)
    random.seed(1)


def __unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def normalize_datapoints(x, factor):
    x = x.astype('float32') / factor
    return x


def random_shuffle(x, y):
    x, y = __unison_shuffled_copies(x, y)
    return x, y


def random_shuffle_and_split(x_train, y_train, x_test, y_test, split_index):
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    x, y = __unison_shuffled_copies(x, y)

    x_train = x[:split_index, :]
    x_test = x[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return (x_train, y_train), (x_test, y_test)
