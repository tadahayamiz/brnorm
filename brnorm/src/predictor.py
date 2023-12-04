# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

@author: tadahaya
"""
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm, trange
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor



class Predictor:
    """ 予測モデル """
    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError


    def predict(self):
        raise NotImplementedError
    

    def eval(self):
        raise NotImplementedError
    

class RFPredictor(Predictor):
    """ Random forest-based predictor """
    def __init__(
            self, seed:int=222, n_estimators:int=100, max_depth:int=6,
            min_samples_split:float=2, min_samples_leaf:int=1,
            max_features:float=0.1,
            oob_score:bool=True, ccp_alpha:float=0.1, 
            max_samples:float=0.9
            ):
        super().__init__()
        self.seed = seed
        self._tmp = RandomForestRegressor(
            random_state=self.seed,
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            oob_score=oob_score, ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            verbose=True
            )
        self.models = []
        self.oob_scores = None


    def fit(self, X, y):
        """
        X: 2d-array
            not different feature matrix
            sample x feature
        
        y: 2d-array
            different feature matrix
            sample x feature
            
        """
        p = y.shape[1]
        self.models = [deepcopy(self._tmp) for i in range(p)]
        self.oob_scores = np.zeros(p)
        for i in trange(p):
            tmp_y = y[:, i]

            print(tmp_y.shape, X.shape)
            
            self.models[i].fit(X, tmp_y)
            self.oob_scores[i] = self.models[i].oob_score


    def predict(self):
        raise NotImplementedError
    

    def eval(self):
        raise NotImplementedError


class ENPredictor(Predictor):
    """ Elastic Net-based predictor """
    def __init__(self):
        super().__init__()
    

    def fit(self):
        raise NotImplementedError


    def predict(self):
        raise NotImplementedError
    

    def eval(self):
        raise NotImplementedError