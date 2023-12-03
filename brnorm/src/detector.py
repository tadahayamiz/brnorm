# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

@author: tadahaya
"""
import pandas as pd
import numpy as np

class Data:
    """ data instance """
    def __init__(self, data=None):
        """
        data: dataframe
            feature x sample (consistent with usual omics data)

        """
        self.X = None # value: sample x feature
        self.sample = None # sample name list
        self.feature = None # feature name list
        self.n_sample = None # sample num
        self.n_feature = None # feature num
        if data is not None:
            self.load(data)


    def load(self, data):
        """
        data: dataframe
            feature x sample (consistent with usual omics data)

        """
        tmp = data.T
        self.X = tmp.values
        self.sample = list(tmp.index)
        self.feature = list(tmp.columns)
        self.n_sample, self.n_feature = tmp.shape


class Detector:
    """
    ground truthとtarget間で異なる遺伝子を見つけてくる
    
    """
    def __init__(self):
        self.grd = Data()
        self.tgt = Data()


    def load_grd(self, data):
        """
        load ground truth data
        
        data: dataframe
            feature x sample matrix
        
        """
        self.grd = Data.load(data)

    
    def load_tgt(self, data):
        """
        load target data
        
        data: dataframe
            feature x sample matrix
        
        """
        self.tgt = Data.load(data)


    def detect(self):
        """ detect differently expressed features """
        raise NotImplementedError


class TwoGroupDetector(Detector):
    """
    detect differently expressed features based on two group test

    """
    def __init__(self):
        super().__init__()
    

    def detect(self):
        """ detect differently expressed features """
        raise NotImplementedError
    

class PCADetector(Detector):
    """
    detect differently expressed features based on PCA

    PCAして異なる因子の上位をとってくる

    """
    def __init__(self):
        super().__init__()
    

    def detect(self):
        """ detect differently expressed features """
        raise NotImplementedError
    

class GroupDetector(Detector):
    """
    detect differently expressed features based on group

    主にTranscriptomeでのTFを想定
    groupベースで差を見つけ, groupに属するものを推定させる

    """
    def __init__(self):
        super().__init__()
    

    def detect(self):
        """ detect differently expressed features """
        raise NotImplementedError
