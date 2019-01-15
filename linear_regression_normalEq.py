#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:52:48 2019

@author: krzysztofrozanski
"""

#Initial packages
import numpy as np
import pandas as pd
import os

# %% Class definition:

class linear_regression_normalEq():
       """Linear regressor.
       Class for implementing linear regression using normal equations.
       
       Parameters
       ----------
       X : {array-like}, shape = (n_samples, n_features)
            Training vectors, where n_samples is the number of 
            samples and n_features is the number of features.
       y : array-like, shape = n_samples
            Response values.
        
       Attributes
       ----------
       theta : 1d-array
        Coefficients after fitting.
        
       """
       def __init__(self, X, y):
           ones = np.array([np.ones((X.shape[0],), dtype=int)])
           self.X = np.concatenate((ones.T, X), axis=1)
           self.y = y
      
       def normalEq(self):
           """ Fits the data using normal equations """
           theta = (np.linalg.pinv((self.X.T @ self.X)) @ self.X.T).dot(self.y)           
           self.theta = theta
           return self


# %% Class initialisation:
        
if __name__ == '__main__':
    
    print('Loading data ...');
    os.chdir('//Users/krzysztofrozanski/Documents/PROGRAMOWANIE/_Machine_learning/machine-learning-ex1/ex1')
    data = pd.read_csv('ex1data2.txt',header = None)
    X = data[[0,1]]
    y = data[2]
     
    print('Running normal equations linear regression ...')
    lr = linear_regression_normalEq(X,y)
    lr.normalEq()
    
    print('Prediction ...')
    p = np.array([1650,3])
    pred = np.dot(p,lr.theta[1:])+ lr.theta[0]
    print(pred)
        
print("Linear regression using normal equations script ends")