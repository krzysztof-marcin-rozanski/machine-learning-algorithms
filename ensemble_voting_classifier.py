"""
Created on Sun June 07 2019

@author: krzysztof_rozanski
"""

# Initial packages
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import time


def ensemble_voting_classifier(estimators, X_df_train, X_df_test, y_train, y_test, voting='hard', only_ensemble=True):
    """ Performs voting classifier for ensemble learning, fitting either
    voting classifier or each member as well """
    voting_clf = VotingClassifier(estimators=estimators, voting=voting)
    if only_ensemble:
        start = time.time()
        print("- Starting:")
        voting_clf.fit(X_df_train, y_train)
        y_preds = voting_clf.predict(X_df_test)
        end = time.time()
        print("- Finished:", end - start)
        print(voting_clf.__class__.__name__, 'Accuracy on test data', accuracy_score(y_test, y_preds))
        print(voting_clf.__class__.__name__, 'Classification report on test data',
              classification_report(y_test, y_preds))
        models = voting_clf
    else:
        clfs = [estimators[i][1] for i in range(len(estimators))]
        clfs.append(voting_clf)
        y_preds = {}
        for clf in clfs:
            start = time.time()
            print("- Starting:")
            clf.fit(X_df_train, y_train)
            y_pred = clf.predict(X_df_test)
            y_preds[clf.__class__.__name__] = y_pred
            end = time.time()
            print("- Finished:", end - start)
            print(clf.__class__.__name__, 'Accuracy on test data', accuracy_score(y_test, y_pred))
            print(clf.__class__.__name__, 'Classification report on test data',
                  classification_report(y_test, y_pred))
        models = clfs
    return models, y_preds
