"""
Created on Thr July 04 2019

@author: krzysztof_rozanski
"""

# Initial packages
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd


def grid_search_cv(model, X_df_train, y_train, param_grid, cv, scoring):
    """ Performs calibration of optimal set of hyperparameters with respect to chosen score,
    using cross-validation. Evaluates all the possible combinations of hyperparameter values.
    Pleas use sklearns metrics.SCORERS.keys() command to select desired score.
    """
    start = time.time()
    print("- Starting:")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, return_train_score=True)
    grid_search.fit(X_df_train, y_train)
    end = time.time()
    print("- Finished:", end-start)
    cvres = grid_search.cv_results_
    print(f"{scoring}_train     ", f"{scoring}_test      ", "Paran:      ")
    for mean_score_train, mean_score_test, params in zip(cvres["mean_train_score"],
                                                         cvres["mean_score_test"],
                                                         cvres["params"]):
        print(mean_score_train, mean_score_test, params)
    return grid_search, pd.DataFrame(cvres)
