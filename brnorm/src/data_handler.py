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
        self.idx_split = idx
        self.y = self.X[idx, :]
        self.idx_remain = [v for v in range(self.n_feature) if v not in idx]
        self.X = self.X[self.idx_remain, :] # update


    def concat(self):
        """ concat again """
        name_split = [self.feature[i] for i in self.idx_split]
        name_remain = [self.feature[i] for i in self.idx_remain]
        spl = pd.DataFrame(self.y, index=name_split, columns=self.sample)
        rem = pd.DataFrame(self.X, index=name_remain, columns=self.sample)
        tmp = pd.concat([rem, spl], axis=0, join="inner")
        tmp = tmp.loc[self.feature, :]
        # update
        self.X = tmp.values
        self.y = None


class DataHandler:
    """ dataのマネージャー"""
    def __init__(self):
        self.grd = Data()
        self.grd_ctl = Data()
        self.tgt = Data()
        self.tgt_ctl = Data()

    
    def get_grd(self):
        """ getter for grd, (whole, control) """
        return (self.grd, self.grd_ctl)


    def get_tgt(self):
        """ getter for tgt, (whole, control) """
        return (self.tgt, self.tgt_ctl)


    def load_grd(self, data, key_ctl:str="dmso", normalize:bool=True):
        """
        load ground truth data
        
        data: dataframe
            feature x sample matrix
        
        """
        ctl = data.loc[:, data.columns.str.contains(key_ctl)]
        if ctl.shape[0] == 0:
            print(f"!! CAUTION: no sample includes {key_ctl} !!")
        ctl_col = list(ctl.columns)
        self.grd.load(data)
        self.grd_ctl.load(data[ctl_col])
        if self.tgt.feature is not None:
            assert self.grd.feature == self.tgt.feature
        if normalize:
            self.grd = (self.grd.T - self.grd.sample_mean) / self.grd.sample_std
            self.grd = self.grd.T
            self.grd_ctl = (self.grd_ctl.T - self.grd_ctl.sample_mean) / self.grd_ctl.sample_std
            self.grd_ctl = self.grd_ctl.T

    
    def load_tgt(self, data, key_ctl:str="dmso", normalize:bool=True):
        """
        load target data
        
        data: dataframe
            feature x sample matrix
        
        """
        ctl = data.loc[:, data.columns.str.contains(key_ctl)]
        if ctl.shape[0] == 0:
            print(f"!! CAUTION: no sample includes {key_ctl} !!")
        ctl_col = list(ctl.columns)
        self.tgt.load(data)
        self.tgt_ctl.load(data[ctl_col])
        if self.grd.feature is not None:
            assert self.grd.feature == self.tgt.feature
        if normalize:
            self.tgt = (self.tgt.T - self.tgt.sample_mean) / self.tgt.sample_std
            self.tgt = self.tgt.T
            self.tgt_ctl = (self.tgt_ctl.T - self.tgt_ctl.sample_mean) / self.tgt_ctl.sample_std
            self.tgt_ctl = self.tgt_ctl.T


    def split_data(self, idx):
        """
        idx: list
            ground truthとtargetで異なるindex
        
        """
        for d in [self.grd_ctl, self.grd, self.tgt_ctl, self.tgt]:
            d.split(idx)


    def concat_data(self):
        """
        idx: list
            ground truthとtargetで異なるindex
        
        """
        for d in [self.grd_ctl, self.grd, self.tgt_ctl, self.tgt]:
            d.concat()