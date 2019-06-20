"""
Created on Thr June 20 2019

@author: krzysztof_rozanski
"""

# Initial packages
from sklearn.metrics import classification_report


def model_classification_report(model, X_df_train, X_df_test, y_train, y_test, obs_out=True):
    """ Fits chosen model into train data and predicts on test data,
    printing sklearns classification report done for 10k sample size
    if obs_out set True, otherwise all test sample size """
    model = model.fit(X_df_train, y_train)
    print(model)
    if obs_out:
        n_out = 10000
        y_pred = model.predict(X_df_test[0:n_out])
        print(classification_report(y_true=y_test[0:n_out], y_pred=y_pred))
    else:
        y_pred = model.predict(X_df_test)
        print(classification_report(y_true=y_test, y_pred=y_pred))
    return model
