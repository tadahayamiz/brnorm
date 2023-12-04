# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

@author: tadahaya
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm, trange

# original
from data_handler import Data

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
        self.grd.load(data)

    
    def load_tgt(self, data):
        """
        load target data
        
        data: dataframe
            feature x sample matrix
        
        """
        self.tgt.load(data)


    def check_feature(self):
        """ check whether features are the same or not"""
        if self.grd.X is None:
            raise ValueError("!! load the ground truth and target data before this !!")
        if self.grd.feature == self.tgt.feature:
            return True
        else:
            return False


    def detect(self):
        """ detect differently expressed features """
        raise NotImplementedError


class TwoGroupDetector(Detector):
    """
    detect differently expressed features based on two group test

    """
    def __init__(self):
        super().__init__()
        self.summary = None
    

    def detect(self, alpha:float=0.05, fold:float=1.0):
        """ detect differently expressed features """
        if not self.check_feature():
            raise ValueError("!! features were not consistent between the ground truth and the target !!")
        p = self.grd.n_feature
        diff = np.zeros((p,))
        pval = np.zeros((p,))
        for i in trange(p):
            pval[i] = stats.ttest_ind(self.grd.X[i], self.tgt.X[i], equal_var=False)[1]
            diff[i] = self.grd.X[i].mean() - self.tgt.X[i].mean()
        qval = multipletests(pval, method="fdr_bh")[1]
        res = pd.DataFrame(
            {"diff":diff, "pval":pval, "qval":qval, "diff_abs":np.abs(diff), "name_feature":self.grd.feature},
            )
        res.loc[:, "check_diff"] = res["diff_abs"] >= fold
        res.loc[:, "check_qval"] = res["qval"] <= alpha
        res.loc[:, "diff_feature"] = res["check_diff"] & res["check_qval"]
        self.summary = res[["diff", "pval", "qval", "name_feature", "diff_feature"]].copy()
        return list(res[res["diff_feature"]].index)


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
