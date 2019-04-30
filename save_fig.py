"""
Created on Tue Apr 01 2019

@author: krzysztof_rozanski
"""

# Initial packages
import os
import matplotlib.pyplot as plt


def save_fig(output_dir, fig_id, tight_layout=True, fig_extension="png"):
    """ Saves image """
    path = os.path.join(output_dir, "images")
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=300)
