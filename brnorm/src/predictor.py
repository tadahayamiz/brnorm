# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

@author: tadahaya
"""
import pandas as pd
import numpy as np

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
    def __init__(self):
        super().__init__()
    

    def fit(self):
        raise NotImplementedError


    def predict(self):
        raise NotImplementedError
    

    def eval(self):
            raise NotImplementedError
