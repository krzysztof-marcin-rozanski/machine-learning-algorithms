"""
Created on Sun July 14 2019

@author: krzysztof_rozanski
"""

# Initial packages
import os
import datetime
import pickle


def model_dump(model, output, date=None):
    """ Dumps the model into pickle format file """
    models_path = os.path.join(output, "models")
    if date:
        model_name = model.__class__.__name__ + '_' + datetime.datetime.today().strftime('%Y-%m-%d')
    else:
        model_name = model.__class__.__name__
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    path = os.path.join(models_path, model_name + ".pk")
    print(f"Saving model into pickle format in {path} ...")
    pickle.dump(model, open(path, 'wb'))
    return print("... done")


def model_load(model_name, output):
    """ Loads the model from pickle format file """
    models_path = os.path.join(output, "models")
    path = os.path.join(models_path, model_name + ".pk")
    print(f"Loading model from pickle format from {path}")
    return pickle.load(open(path, 'rb'))


def pickle_dump(obj, path, file):
    """ Dumps the python object, i.e. dataframe or model into pickle format file """
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, file + ".pk")
    print(f"Saving object into pickle format in {path} ...")
    pickle.dump(obj, open(path, 'wb'))
    return print("... done")


def pickle_load(file, path):
    """ Loads the python object, i.e. dataframe or model from pickle format file """
    path = os.path.join(path, file + ".pk")
    print(f"Loading object from pickle format from {path}")
    return pickle.load(open(path, 'rb'))
