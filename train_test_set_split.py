"""
Created on Sun May 06 2019

@author: krzysztof_rozanski
"""

# Initial packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_test_set_split(X_df, y_target, test_size, random_state=42, normalize=True):
    """ Uses sklearn to create split of sets, then normalizes the train set and transforms
    the test set with training sets normalization parameters """
    X_df_train, X_df_test, y_train, y_test = train_test_split(X_df, y_target, test_size=test_size, random_state=random_state)
    X_train_mean, X_train_std = None, None
    if normalize:  # Data normalization
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_df_train = scaler.fit_transform(X_df_train)
        X_train_mean, X_train_std = scaler.mean_, scaler.var_
        X_df_test = scaler.transform(X_df_test)
    return X_df_train, X_df_test, y_train, y_test, X_train_mean, X_train_std
