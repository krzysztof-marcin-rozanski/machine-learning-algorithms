"""
Created on Thr July 05 2019

@author: krzysztof_rozanski
"""

# Initial packages
from sklearn.model_selection import RandomizedSearchCV
import time
import pandas as pd


def randomized_search_cv(model, X_df_train, y_train, param_distribs, n_iter, cv, scoring, n_jobs=4):
    """ Performs calibration of optimal set of hyperparameters with respect to chosen score,
    using cross-validation. Evaluates given number of random combinations by selecting a random
    value for each hyperparameter at every iteration.
    Pleas use sklearns metrics.SCORERS.keys() command to select desired score.
    """
    start = time.time()
    print("- Starting:")
    rnd_search = RandomizedSearchCV(estimator=model, param_distributions=param_distribs,
                                    n_iter=n_iter, cv=cv, scoring=scoring, return_train_score=True, n_jobs=n_jobs)
    rnd_search.fit(X_df_train, y_train)
    end = time.time()
    print("- Finished:", end - start)
    cvres = rnd_search.cv_results_
    print(f"{scoring}_train     ", f"{scoring}_test      ", "Param:      ")
    for mean_score_train, mean_score_test, params in zip(cvres["mean_train_score"],
                                                         cvres["mean_score_test"],
                                                         cvres["params"]):
        print(mean_score_train, mean_score_test, params)
    return rnd_search, pd.DataFrame(cvres)
