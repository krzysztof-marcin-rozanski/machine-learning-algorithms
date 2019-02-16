#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:26:38 2019

@author: krzysztofrozanski
"""

#Initial packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# %% Class definition:

class logistic_regression_batchGD():
       """Logistic regressor.
       Class for implementing logistic regression with batch gradient descent
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
            Binary response values.
        
       Attributes
       ----------
       theta : 1d-array
        Coefficients after fitting.
       costs : list
        List of cost function value in every epoch.
        
       """
       def __init__(self, alpha=0.001, epochs=100):
           self.alpha = alpha
           self.epochs = epochs
          
       def __call__(self, X, y):
           self.X = np.c_[np.ones((X.shape[0],1)), X]
           self.y = y
       
       def batch_gd(self):
           """ Fits the data using batch gradient descent algorithms """
           theta = np.zeros(self.X.shape[1])
           costs = []
           for epoch in range(self.epochs):
                theta = theta - self.alpha * logistic_regression_batchGD.cost_function(self.X, self.y, theta)[1]
                cost = logistic_regression_batchGD.cost_function(self.X, self.y, theta)[0]
                print (f'Cost function value is {cost}.')
                costs.append(cost)
           self.theta = theta
           self.costs = costs
           return self 
        
       def predict(self, x):
           """ Return class label after applying sigmoid function and unit step function """  
           return np.where(logistic_regression_batchGD.sigmoid(x.dot(self.theta)) >= 0.5, 1, 0) 
            
       @staticmethod
       def cost_function(X, y, theta):
           """ Computes the cost of using theta as the parameter for 
               logistic regression to fit the data points in X and y """
           n = len(y)
           h = logistic_regression_batchGD.sigmoid(X.dot(theta))
           grad = (1/n) * X.T.dot(h - y)
           J = - (1/n) * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h)))
           return J, grad

       @staticmethod
       def sigmoid(z):
           """ Return the output of the logistic/sigmoid function """
           return 1.0 / (1.0 + np.exp(-z))
      
    
# %% Class initialisation:
        
if __name__ == '__main__':
    
    print('Loading data ...');
    os.chdir('//Users/krzysztofrozanski/Documents/PROGRAMOWANIE/_Machine_learning/machine-learning-ex2/ex2')
    data = pd.read_csv('ex2data1.txt',header = None)
    X = data[[0,1]]
    y = data[2]
     
    print('Running batch gradient descent ...')
    lr = logistic_regression_batchGD(0.001, 100000)
    lr(X,y)
    lr.cost_function(lr.X,lr.y,theta = np.zeros(lr.X.shape[1]))
    lr.cost_function(lr.X,lr.y,[-24, 0.2, 0.2])
    lr.batch_gd()
    plt.plot(range(len(lr.costs)),lr.costs)
    lr.cost_function(lr.X,lr.y,lr.theta)
    
    print('Prediction ...')
    prob = lr.sigmoid(np.dot(np.array([1, 45, 85]), np.array([-25.16127,   0.20623,   0.20147])))
    pred = lr.predict(lr.X)
    print(pred)
    print('Train Accuracy:', np.mean((pred == y)) * 100)
        
print("Linear regression with batch gradient descent script ends")