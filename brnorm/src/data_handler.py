# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

@author: tadahaya
"""
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange

class Data:
    """ data instance """
    def __init__(self, data=None):
        """
        data: dataframe
            feature x sample (consistent with usual omics data)

        """
        self.X = None # value: feature x sample
        self.sample = None # sample name list
        self.feature = None # feature name list
        self.n_sample = None # sample num
        self.n_feature = None # feature num
        self.sample_mean = None # samplewise mean
        self.sample_std = None # samplewise std
        self.y = None # value: split feature x sample
        self.idx_split = []
        self.idx_remain = []
        if data is not None:
            self.load(data)


    def load(self, data):
        """
        data: dataframe
            feature x sample (consistent with usual omics data)

        """
        self.X = data.values
        self.sample = list(data.columns)
        self.feature = list(data.index)
        self.n_feature, self.n_sample = data.shape
        self.sample_mean = data.mean(axis=0)
        self.sample_std = data.std(axis=0, ddof=1)


    def split(self, idx):
        """ split data based on the given index """
        self.idx_remain = idx
        self.y = self.X[idx, :]
        self.idx_remain = [v for v in range(self.n_feature) if v not in idx]
        self.X = self.X[self.idx_remain, :] # update


    def concat(self):
        """ concat again """
        tmp = self.zeros((self.n_feature, self.n_sample))
        for i in range(self.n_feature):
            if i in self.idx_remain:
                tmp[i, :] = self.X[0]
                del self.X[0]
            elif i in self.idx_split:
                tmp[i, :] = self.y[0]
                del self.y[0]
            else:
                raise IndexError
        assert self.X.shape[0] == 0
        self.X = tmp


class DataHandler:
    """ dataのマネージャー"""
    def __init__(self):
        self.grd_trt = Data()
        self.grd_ctl = Data()
        self.tgt_trt = Data()
        self.tgt_ctl = Data()

    
    def load_grd(self, data, key_ctl:str="dmso"):
        """
        load ground truth data
        
        data: dataframe
            feature x sample matrix
        
        """
        ctl = data.loc[:, data.columns.str.contains(key_ctl)]
        ctl_col = list(ctl.columns)
        trt_col = [v for v in data.columns if v not in ctl_col]
        self.grd_trt.load(data[trt_col])
        self.grd_ctl.load(data[ctl_col])

    
    def load_tgt(self, data, key_ctl:str="dmso"):
        """
        load target data
        
        data: dataframe
            feature x sample matrix
        
        """
        ctl = data.loc[:, data.columns.str.contains(key_ctl)]
        ctl_col = list(ctl.columns)
        trt_col = [v for v in data.columns if v not in ctl_col]
        self.tgt_trt.load(data[trt_col])
        self.tgt_ctl.load(data[ctl_col])


    def split_data(self, idx):
        """
        idx: list
            ground truthとtargetで異なるindex
        
        """
        for d in [self.grd_ctl, self.grd_trt, self.tgt_ctl, self.tgt_trt]:
            pass
