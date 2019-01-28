#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 23:10:25 2019

@author: krzysztofrozanski
"""
import numpy as np
import pandas as pd


def map_features(x1, x2, degree = 6):
    out = pd.DataFrame({'1':np.ones((x1.shape[0],), dtype=int)})
    for i in range(1,degree +1):
        for j in range(0,i + 1):
            out[f'x1^{i-j},x2^{j}'] = x1**(i-j) * x2**(j)
            print(i-j,j)
    return out

x = map_features(x1,x2,degree = 3)
