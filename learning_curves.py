"""
Created on Thr June 20 2019

@author: krzysztof_rozanski
"""

# Initial packages
import time
import matplotlib.pyplot as plt
from save_fig import save_fig


def learning_curves(model, X_df_train, X_df_test, y_train, y_test, score, multiplier=10000, plot_cat=True, output=None):
    """ For given score (i.e. MSE, precision, recall or f1 score) fits the model to growing number of
    data and plots the training and test performance curves. Please use sklearns metrics.SCORERS.keys()
    command to select desired score """
    start = time.time()
    print("- Starting:")
    train_score, val_score, val_dict = [], [], {}
    cats = list(y_test.unique())
    for cat in cats:
        val_dict[cat] = []
    for m in range(1, int(len(X_df_train)/multiplier)):
        model.fit(X_df_train[:m * multiplier], y_train[:m * multiplier])
        y_train_predict = model.predict(X_df_train[:m * multiplier])
        y_val_predict = model.predict(X_df_test)
        train_score.append(score(y_train[:m * multiplier], y_train_predict, average='weighted'))
        val_score.append(score(y_test, y_val_predict, average='weighted'))
        if plot_cat:
            for cat in cats:
                val_dict[cat].append(score(y_test, y_val_predict, average='weighted', labels=[cat]))
    end = time.time()
    print("- Finished:", end-start)
    plt.plot(train_score, "r-+", linewidth=2, label="train_weighted_avg")
    plt.plot(val_score, "b-+", linewidth=3, label="test_weighted_ang")
    if plot_cat:
        for cat in cats:
            plt.plot(val_dict[cat], linewidth=3, label=cat)
    plt.legend(loc="lower right", fontsize=8)
    plt.xlabel(f"{model.__class__.__name__}, training set size (in {multiplier}'s)", fontsize=14)
    plt.ylabel(score, fontsize=14)
    save_fig(output_dir=output, fig_id=f"{model.__class__.__name__} learning curves plot")
    plt.show()
    return train_score, val_score, val_dict
