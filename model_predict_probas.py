"""
Created on Mon June 08 2019

@author: krzysztof_rozanski
"""

# Initial packages
import pandas as pd
import numpy as np


def model_predict_probas(model, X):
    """ Predicts models first and second classification choice, i.e. normal prediction
     and then second most probable one """
    model_predicted_prob = pd.DataFrame(model.predict_proba(X), columns=model.classes_)
    model_predicted_prob_class = model_predicted_prob.copy()
    model_predicted_prob_class['Classification_first'] = model_predicted_prob.apply(ProbaProcessor('1'), axis=1)
    model_predicted_prob_class['Classification_second'] = model_predicted_prob.apply(ProbaProcessor('2'), axis=1)
    return model_predicted_prob_class


class ProbaProcessor:
    def __init__(self, index):
        self.index = index

    def __call__(self, row):
        if self.index == '1':
            label = np.argmax(row)
        elif self.index == '2':
            if np.max(row) == 1:
                label = None
            else:
                label = np.argmax(row.drop(np.argmax(row)))
        else:
            label = np.argmax(row)
        return label
