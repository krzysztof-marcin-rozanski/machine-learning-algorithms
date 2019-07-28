"""
Created on Sun July 28 2019

@author: krzysztof_rozanski
"""

# Initial packages
from save_fig import save_fig
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from scipy import stats


def kernel_plot(file_path, var1):
    """ Plots kernel density """
    rcParams['figure.figsize'] = 12, 8
    plt.style.use('seaborn-white')
    sns.distplot(var1, hist=True, kde=True, color='blue',
                 kde_kws={'color': 'orange', 'alpha': 0.8},
                 hist_kws={'alpha': 0.6})
    plt.title(f'Kernel density plot of {var1.name} variable')
    plt.tight_layout()
    save_fig(file_path, f'Kernel plot of {var1.name}')
    return plt.show()


def scatter_plot(file_path, var1, var2):
    """ Plots scatter plot """
    plt.style.use('seaborn-darkgrid')
    plt.scatter(var1, var2)
    plt.xlabel(var1.name)
    plt.ylabel(var2.name)
    plt.tight_layout()
    save_fig(file_path, f'Scatter plot of {var1.name} and {var2.name}')
    return plt.show()


def marginal_kernel_plot(file_path, data, var1, var2):
    """ Plots marginal kernel density """
    plt.style.use('seaborn-white')
    sns.jointplot(x=var1, y=var2, data=data,
                  kind='kde', size=8, space=0)
    plt.tight_layout()
    save_fig(file_path, f'Marginal kernel plot of {var1} and {var2} variables')
    return plt.show()


def var_normality_stats(file_path, var):
    """ Plots Q-Q plot and test normality using Shapiro-Wilk test """
    stats.probplot(var, plot=plt)
    plt.tight_layout()
    save_fig(file_path, f'Q-Q plot of {var.name}')
    shapiro = stats.shapiro(var)
    print(' Shapiro-Wilk test for normality: ', shapiro)
    if shapiro[1] < 0.05:
        print('Distribution of variable is not normal.')
    else:
        print('Cannot reject null hypthesis about normality.')
    return None


def plot_residuals(file_path, residuals, model):
    """ Plots residuals and tests normality"""
    rcParams['figure.figsize'] = 12, 8
    plt.plot('Error', data=residuals, marker='o', ls='')
    plt.title(f'Scatterplot of {model} residuals')
    plt.tight_layout()
    save_fig(file_path, f'Scatterplot of {model} residuals')
    shapiro = stats.shapiro(residuals)
    print(' Shapiro-Wilk test for normality: ', shapiro)
    if shapiro[1] < 0.05:
        print('Distribution of residuals is not normal.')
    else:
        print('Cannot reject null hypthesis about residuals normality.')
    return None


def plot_residuals_histogram(file_path, residuals, model):
    """ Plots histogram of residuals"""
    plt.hist('Error', bins=30, density=True, data=residuals)
    plt.title(f'Histogram of {model} residuals')
    plt.tight_layout()
    save_fig(file_path, f'Histogram of {model} residuals')
    return None


def reg_scatter_plot(file_path, data, var1, var2):
    """ A scatterplot with a regression line """
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.regplot(x=var1, y=var2, data=data, fit_reg=True,
                scatter_kws={'marker': 'D', 'color': 'blue', 'alpha': 0.6},
                line_kws={'color': 'orange', 'alpha': 0.7})
    plt.title('The relationship between x1_1x2_1 and Y',
              fontdict={'fontsize': 18, 'fontweight': 'bold'})
    plt.xlabel('x1_1x2_1', **{'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel('Y', **{'fontsize': 14, 'fontweight': 'bold'})
    plt.tight_layout()
    save_fig(file_path, f'Scatterplot with a regression line')
    return None
