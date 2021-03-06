'''
Module for selecting features
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.models.train_model import get_features_and_labels
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import RFE


def select_features_efs(data, clf, scoring='accuracy', seed=1234, n_features=5):
    '''
    :param data: dataframe (with last column as outcome)
    :param clf: sklearn (for example LogisticRegression()
    :param scoring: metric for evaluating
    :param seed: random state
    :return: array of indices for  selected variables
    '''

    X_train, X_test, y_train, y_test = get_features_and_labels(data_tmp, seed)

#    X_train, y_train = data.iloc[:, :-1].values, data.iloc[:, -1].values
    efs1 = EFS(clf,
               min_features=n_features,
               max_features=n_features,
               scoring=scoring,
               print_progress=True,
               cv=0,
               n_jobs=-1)
    efs1 = efs1.fit(X_train, y_train)

    print('\nSelected features by EFS:', data.columns[list(efs1.best_idx_)])

    X_train_efs = efs1.transform(X_train)
    X_test_efs = efs1.transform(X_test)

    score = LogisticRegression().fit(X_train, y_train).score(X_test, y_test)
    print("\nПравильность на полном наборе(EFS): {:.3f}".format(score))

    score = LogisticRegression().fit(X_train_efs, y_train).score(X_test_efs, y_test)
    print("\nПравильность на тестовом наборе(EFS): {:.3f}".format(score))

    return list(efs1.best_idx_)



def select_features_sfs(data, clf, k_features=5, forward=True, floating=False, scoring='accuracy',  verbose=2,
                        cv=5, seed=1234):
    '''

    :param data:
    :param clf:
    :param k_features:
    :param forward:
    :param floating:
    :param scoring:
    :param verbose:
    :param cv:
    :param seed:
    :return:
    '''


    X_train, X_test, y_train, y_test = get_features_and_labels(data, seed)

    from mlxtend.feature_selection import SequentialFeatureSelector as SFS

    sfs1 = SFS(clf,
           k_features=k_features,
           forward=forward,
           floating=floating,
           verbose=verbose,
           scoring=scoring,
           cv=cv,
           n_jobs=-1)

    sfs1 = sfs1.fit(X_train, y_train)

    print('\nSelected features by SFS:', data.columns[list(sfs1.k_feature_idx_)])

    X_train_sfs = sfs1.transform(X_train)
    X_test_sfs = sfs1.transform(X_test)

    score = LogisticRegression().fit(X_train, y_train).score(X_test, y_test)
    print("\nПравильность на полном наборе(SFS): {:.3f}".format(score))

    score = LogisticRegression().fit(X_train_sfs, y_train).score(X_test_sfs, y_test)
    print("\nПравильность на тестовом наборе(SFS): {:.3f}".format(score))

    return list(sfs1.k_feature_idx_)



def select_features_rfe(data, clf, scoring='accuracy', seed=1234, n_features=5):
    '''
    :param data: dataframe (with last column as outcome)
    :param clf: sklearn (for example LogisticRegression()
    :param scoring: metric for evaluating
    :param seed: random state
    :return: array of indices for  selected variables
    '''


    X_train, X_test, y_train, y_test = get_features_and_labels(data, seed)

    select = RFE(clf,  n_features_to_select=n_features)

    select.fit(X_train,
               y_train)
    # визуализируем отобранные признаки :
    mask = select.get_support(indices=True)
    print('\nSelected features by RFE:', data.columns[mask])

    X_train_rfe = select.transform(X_train)
    X_test_rfe = select.transform(X_test)

    score = LogisticRegression().fit(X_train, y_train).score(X_test, y_test)
    print("\nПравильность на полном наборе(RFE): {:.3f}".format(score))

    score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
    print("\nПравильность на тестовом наборе(RFE): {:.3f}".format(score))

    return list(mask)




if __name__ == '__main__':
    import os

    default_path = os.getcwd()
    print(default_path)  # Prints the working directory
    #path = "D:/Python-projects/LoanPrediction3/"
    #os.chdir(path)

    data_tmp = pd.read_csv('../../data/interim/train_ready.csv', index_col=0)
    seed = 12345

    # clf = LogisticRegression()
    # clf = KNeighborsClassifier(n_neighbors=4)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)

    # pos = list(select_features_efs(data_tmp, clf, scoring='accuracy',n_features=10))
    # features = data_tmp.columns[pos]

    n_fts = 10
    pos = select_features_sfs(data_tmp,
                              clf,
                              k_features=5,
                              forward=True,
                              floating=False,
                              scoring='accuracy',
                              verbose=2,
                              cv=5,
                              seed=1234)
    features = data_tmp.columns[pos]

    #pos = select_features_rfe(data_tmp, clf, scoring='accuracy', n_features=n_fts))
    #features = data_tmp.columns[pos]
