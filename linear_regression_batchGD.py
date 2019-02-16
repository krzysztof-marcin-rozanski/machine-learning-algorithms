#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 16:42:23 2019

@author: krzysztofrozanski
"""

#Initial packages
import numpy as np
import pandas as pd
import os

# %% Class definition:

class linear_regression_batchGD():
       """Linear regressor.
       Class for implementing linear regression with batch gradient descent
       learning algorithm.
       
       Parameters
       ----------
       alpha : float
        Learning rate (between 0.0 and 1.0)
       epochs : int
        Iterations of the training dataset.
       X : {array-like}, shape = (n_samples, n_features)
            Training vectors, where n_samples is the number of 
            samples and n_features is the number of features.
       y : array-like, shape = n_samples
            Response values.
        
       Attributes
       ----------
       theta : 1d-array
        Coefficients after fitting.
       costs : list
        List of cost function value in every epoch.
        
       """
       def __init__(self, alpha=0.01, epochs=100):
           self.alpha = alpha
           self.epochs = epochs
    
       def __call__(self, X, y):        
           self.X = np.c_[np.ones((X.shape[0],1)), X]
           self.y = y
      
       def batch_gd(self):
           """ Fits the data using batch gradient descent algorithms """
           theta = np.zeros(self.X.shape[1])
           costs = []
           n = len(self.y)
           for epoch in range(self.epochs):
                theta = theta - self.alpha * (1/n) * np.dot(self.X.T,(self.X.dot(theta) - self.y))
                cost = linear_regression_batchGD.cost_function(self.X, self.y, theta)
                print (f'Cost function value is {cost}.')
                costs.append(cost)
           self.theta = theta
           self.costs = costs
           return self
        
       @staticmethod
       def cost_function(X, y, theta):
           """ Computes the cost of using theta as the parameter for 
               linear regression to fit the data points in X and y """
           n = len(y)
           return 1/(2 * n) * (X.dot(theta) - y).T @ (X.dot(theta) - y)


# %% Class initialisation:
        
if __name__ == '__main__':
    
    print('Loading data ...');
    os.chdir('//Users/krzysztofrozanski/Documents/PROGRAMOWANIE/_Machine_learning/machine-learning-ex1/ex1')
    data = pd.read_csv('ex1data2.txt',header = None)
    X = data[[0,1]]
    y = data[2]
    
    print('Normalizing features ...')
    X, mu, sigma = (X - np.mean(X, axis=0))/np.std(X, axis=0),np.mean(X, axis=0),np.std(X, axis=0)
    
    print('Running batch gradient descent ...')
    lr = linear_regression_batchGD(0.01, 400)
    lr(X,y)
    lr.batch_gd()
    
    print('Prediction ...')
    p = np.array([1650,3])
    p_tr = (p - mu)/sigma
    pred = np.dot(p_tr,lr.theta[1:]) + lr.theta[0]
    print(pred)
        
print("Linear regression with batch gradient descent script ends")