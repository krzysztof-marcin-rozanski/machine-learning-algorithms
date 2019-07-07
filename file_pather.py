"""
Created on Wed May 01 2019

@author: krzysztof_rozanski
"""

# Initial packages
import os
import sys


def file_pather(file_path=None):
    """ Changes the current working directory """
    if not file_path:
        os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
    else:
        os.chdir(file_path)
    return print(os.getcwd())
