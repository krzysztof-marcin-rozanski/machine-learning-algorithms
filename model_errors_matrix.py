"""
Created on Sat July 06 2019

@author: krzysztof_rozanski
"""

# Initial packages
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from save_fig import save_fig


def model_errors_matrix(model, X, y, output_dir=None):
    """ Plots confusion matrix and non-diagonal errors """
    y_pred = model.predict(X)
    conf_mx = confusion_matrix(y_true=y, y_pred=y_pred)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.xlabel(f"{model.__class__.__name__}'s confusion matrix plot", fontsize=14)
    save_fig(output_dir, f"{model.__class__.__name__}'s confusion matrix plot", tight_layout=False)
    plt.show()
    row_sums = conf_mx(axis=1, keepdims=True)
    norm_conf_mx = conf_mx/row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.xlabel(f"{model.__class__.__name__}'s confusion matrix errors plot", fontsize=14)
    save_fig(output_dir, f"{model.__class__.__name__}'s confusion matrix errors plot", tight_layout=False)
    plt.show()
    return conf_mx, norm_conf_mx, list(y.sort_values().unique())
