import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm

##################################################################################################
# First things first! Set a seed for reproducibility.
# https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def __get_device__() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print('Device available is', device)
    return device

## Shuffle and pick a quarter of the data
def __shuffle_pick_quarter_data__ (x_f, y_f, x_s, y_s) :
    # Take only quarter dataset
    random.shuffle(x_f)
    random.shuffle(x_s)
    x_s_h = x_s[0:int(len(x_s)/4)]
    y_s_h = y_s[0:int(len(x_s)/4)]
    
    
    # Comment this for unblanced data
    # Stutter is less than fluent
    x_f_h = x_f[0:len(x_s_h)]
    y_f_h = y_f[0:len(x_s_h)]
    
    x_train = x_s_h + x_f_h
    y_train = y_s_h + y_f_h
    return x_train, y_train

## Shuffle and pick half of the data
def __shuffle_pick_quarter_data__ (x_f, y_f, x_s, y_s) :
    # Take only quarter dataset
    random.shuffle(x_f)
    random.shuffle(x_s)
    x_s_h = x_s[0:int(len(x_s)/2)]
    y_s_h = y_s[0:int(len(x_s)/2)]
    
    
    # Comment this for unblanced data
    # Stutter is less than fluent
    x_f_h = x_f[0:len(x_s_h)]
    y_f_h = y_f[0:len(x_s_h)]
    
    x_train = x_s_h + x_f_h
    y_train = y_s_h + y_f_h
    return x_train, y_train


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

##################################################################################################
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter():
    def __init__(self, num_batches, meters, prefix=""): 
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'