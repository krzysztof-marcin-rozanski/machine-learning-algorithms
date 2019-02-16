#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:09:16 2019

@author: krzysztofrozanski
"""

#Initial packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from map_features import map_features

# %% Class definition:

class logistic_regression_batchGD_regularization():
       """Logistic regressor regularized.
       Class for implementing regularized logistic regression with batch gradient descent
       learning algorithm.
       
       Parameters
       ----------
       alpha : float
        Learning rate (between 0.0 and 1.0)
       epochs : int
        Iterations of the training dataset.
       lambda : float
        Regularization/penalty parameter.
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
       def __init__(self, alpha=0.01, epochs=100, lambda_=1.0):
           self.alpha = alpha
           self.epochs = epochs
           self.lambda_ = lambda_
    
       def __call__(self, X, y):
           self.X = X
           self.y = y
           
       def batch_gd(self):
           """ Fits the data using batch gradient descent algorithms """
           theta = np.zeros(self.X.shape[1])
           costs = []
           for epoch in range(self.epochs):
                theta = theta - self.alpha * logistic_regression_batchGD_regularization.cost_function(self.X, self.y, theta, self.lambda_)[1]
                cost = logistic_regression_batchGD_regularization.cost_function(self.X, self.y, theta, self.lambda_)[0]
                print (f'Cost function value is {cost}.')
                costs.append(cost)
           self.theta = theta
           self.costs = costs
           return self 
        
       def predict(self, x):
           """ Return class label after applying sigmoid function and unit step function """
           return np.where(logistic_regression_batchGD_regularization.sigmoid(x.dot(self.theta)) >= 0.5, 1, 0) 
            
       @staticmethod
       def cost_function(X, y, theta, lambda_):
           """ Computes the cost of using theta as the parameter for 
               logistic regression to fit the data points in X and y """
           n = len(y)
           h = logistic_regression_batchGD_regularization.sigmoid(X.dot(theta))
           grad = np.array(np.zeros(X.shape[1]))
           grad = (1/n) * X.T.dot(h - y)
           temp = theta
           temp[0] = 0.0 #because we don't want to regularize bias term
           grad += lambda_/n * temp
           J = - (1/n) * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h))) + lambda_/(2*n) * temp.dot(temp)
           return J, grad

       @staticmethod
       def sigmoid(z):
           """ Return the output of the logistic/sigmoid function """
           return 1.0 / (1.0 + np.exp(-z))


# %% Class initialisation:
        
if __name__ == '__main__':
    
    print('Loading data ...')
    os.chdir('//Users/krzysztofrozanski/Documents/PROGRAMOWANIE/_Machine_learning/machine-learning-ex2/ex2')
    data = pd.read_csv('ex2data2.txt',header = None)
    X = data[[0,1]]
    y = data[2]
    
    print('Maping the features into polynomial ...')
    X = map_features(X[0],X[1], degree = 6)
    
    print('Running batch gradient descent ...')
    lr = logistic_regression_batchGD_regularization(0.01, 10000, 1.0)
    lr(X,y)
    lr.batch_gd()  
    
    print('Prediction ...')
    pred = lr.predict(lr.X)
    print(pred)
    print('Train Accuracy:', np.mean((pred == y)) * 100)
        
print("Regularized logistic regression with batch gradient descent script ends")
    
    
    
    