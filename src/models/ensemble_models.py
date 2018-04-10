# A host of Scikit-learn models
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import os
mingw_path = 'C:\\Program Files\\Git\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline


def get_models():
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)
    xg = xgb.XGBClassifier(max_depth=4, learning_rate=0.1,n_estimators=100,
                                         subsample=0.5,
                                         colsample_bytree=0.5,
                                         nthread=-1,
                                         seed=SEED)
    lg = lgb.LGBMClassifier(num_leaves=4, learning_rate=0.07, n_estimators=1000,
                                          colsample_bytree=0.5, subsample=0.5,
                                          nthread=-1, random_state=SEED)

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              'xgb': xg,
              'lightGBM': lg
              }

    return models


def train_predict(model_list, xtrain, ytrain, xtest, ytest):
    """Fit models in list on training set and return preds
    model_list - list of ensembled models,
    xtrain - train set,
    ytrain - train outcome variable,
    xtest - test set
    ytest - test outcome variable,
    """
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(xtrain, ytrain)
        P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")


from sklearn.metrics import roc_curve
def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    cm = [plt.cm.rainbow(i)
          for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]

    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    fpr, tpr, _ = roc_curve(ytest, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.show()



# Examples
if __name__ == "__main__":
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.model_selection import cross_val_score, cross_val_predict
    # We will calculate the P-R curve for each classifier
    from sklearn.metrics import roc_auc_score, roc_curve
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    SEED = 7
    scoring = 'roc_auc'
    kfold = StratifiedKFold(n_splits=10, random_state=SEED)

    data_tmp = pd.read_csv('../../data/processed/train_ready.csv', index_col=0)
    trainSet, testSet, y_train, y_test = get_features_and_labels(data_tmp)
    models = get_models()
    P = train_predict(models, trainSet,y_train,testSet,y_test)
    score_models(P, y_test)

    plot_roc_curve(y_test, P.values, P.mean(axis=1), list(P.columns), "ensemble")