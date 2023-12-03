# -*- coding: utf-8 -*-
"""

not CLI package

@author: tadahaya
"""
# packages installed in the current environment
import os
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import glob

# original packages in src
from .src import utils
from .src import detector
from .src import predictor

SEP = os.sep

class BRNorm:
    def __init__(self):
        pass


    def main(self):
        # 差があるトップKを抜いてくる
        ## control群に絞る？
        # detector.detect()
        
        # ground truth側でトップKを推定するモデルを構築
        # predcitor.fit()
        
        # target側で2.のモデルを使ってトップKを推定
        # target側のトップKを推定値で置き換える
        # predcitor.predict()

        # 先のトップKについて, RMSE
        # predictor.eval()

        
        pass