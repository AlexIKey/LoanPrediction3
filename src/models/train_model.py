import matplotlib
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import os
mingw_path = 'C:\\Program Files\\Git\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
#os.environ['XGBOOST_BUILD_DOC'] = 'D:\\Users\\Alex\\Anaconda3\\envs\\datascience\\Lib\\site-packages\\xgboost;'
import xgboost as xgb

class djLGB(BaseEstimator, ClassifierMixin):
    """смесь lgb и xgb"""

    def __init__(self, seed=0, nest_lgb=1.0, nest_xgb=1.0, cbt=0.5, ss=0.5, alpha=0.5):
        """
        Инициализация
        seed - инициализация генератора псевдослучайных чисел
        nest_lgb, nest_xgb - сколько деревьев использовать (множитель)
        cbt, ss - процент признаков и объектов для сэмплирования
        alpha - коэффициент доверия XGB
        """
        print('LGB + XGB')
        self.models = [lgb.LGBMClassifier(num_leaves=2, learning_rate=0.07, n_estimators=int(1400 * nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=0 + seed),
                       lgb.LGBMClassifier(num_leaves=3, learning_rate=0.07, n_estimators=int(800 * nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=1 + seed),
                       lgb.LGBMClassifier(num_leaves=4, learning_rate=0.07, n_estimators=int(800 * nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=2 + seed),
                       lgb.LGBMClassifier(num_leaves=5, learning_rate=0.07, n_estimators=int(600 * nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=3 + seed, ),
                       xgb.XGBClassifier(max_depth=1,
                                         learning_rate=0.1,
                                         n_estimators=int(800 * nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=0 + seed),
                       xgb.XGBClassifier(max_depth=2,
                                         learning_rate=0.1,
                                         n_estimators=int(400 * nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=1 + seed),
                       xgb.XGBClassifier(max_depth=3,
                                         learning_rate=0.1,
                                         n_estimators=int(200 * nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=2 + seed),
                       xgb.XGBClassifier(max_depth=4,
                                         learning_rate=0.1,
                                         n_estimators=int(100 * nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=3 + seed)
                       ]
        self.weights = [(1 - alpha) * 1, (1 - alpha) * 1, (1 - alpha) * 1, (1 - alpha) * 0.5, alpha * 0.5, alpha * 1,
                        alpha * 1.5, alpha * 0.5]
        #print(self.models)

    def fit(self, X, y=None):
        """
        обучение
        """
        for t, clf in enumerate(self.models):
            print('train', t)
            clf.fit(X, y)
        return self

    def predict(self, X):
        """
        определение вероятности
        """
        suma = 0.0
        for t, clf in enumerate(self.models):
            a = clf.predict_proba(X)[:, 1]
            suma += (self.weights[t] * a)
        return (suma / sum(self.weights))

    def predict_proba(self, X):
        """
        определение вероятности
        """
        return (self.predict(X))

# Examples
if __name__ == "__main__":
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.model_selection import cross_val_score, cross_val_predict
    # We will calculate the P-R curve for each classifier
    from sklearn.metrics import  roc_auc_score, roc_curve
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt



    def get_features_and_labels(frame):
        '''
        Transforms and scales the input data and returns numpy arrays for
        training and testing inputs and targets.
        '''

        # Replace missing values with 0.0, or we can use
        # scikit-learn to calculate missing values (below)
        # frame[frame.isnull()] = 0.0

        # Convert values to floats
        arr = np.array(frame, dtype=np.float)

        # Use the last column as the target value
        X, y = arr[:, :-1], arr[:, -1]
        # To use the first column instead, change the index value
        # X, y = arr[:, 1:], arr[:, 0]

        # Use 80% of the data for training; test against the rest
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # sklearn.pipeline.make_pipeline could also be used to chain
        # processing and classification into a black box, but here we do
        # them separately.

        # If values are missing we could impute them from the training data
        # from sklearn.preprocessing import Imputer
        # imputer = Imputer(strategy='mean')
        # imputer.fit(X_train)
        # X_train = imputer.transform(X_train)
        # X_test = imputer.transform(X_test)

        # Normalize the attribute values to mean=0 and variance=1
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # To scale to a specified range, use MinMaxScaler
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit the scaler based on the training data, then apply the same
        # scaling to both training and test sets.
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Return the training and test sets
        return X_train, X_test, y_train, y_test


    seed = 7
    scoring = 'roc_auc'
    kfold = StratifiedKFold(n_splits=10, random_state=seed)

    # import os;
    # default_path = os.getcwd()
    # print(default_path);  # Prints the working directory
    # #os.chdir(path)

    data_tmp = pd.read_csv('../../data/processed/train_ready.csv', index_col=0)
    trainSet, testSet, y_train, y_test = get_features_and_labels(data_tmp)

    model = djLGB(seed=2000, nest_lgb=1.3, nest_xgb=1.3)
    name = 'djLGB'
    model.fit(trainSet, y_train)

    y_pred = cross_val_predict(djLGB(seed=2000, nest_lgb=1.3, nest_xgb=1.3), trainSet, y_train, cv=kfold)
    #auc = roc_auc_score( y_test.astype(int), y_pred.astype(int))
    #print("AUC_CV:",auc)
    #cv_results = cross_val_score(model, trainSet, y_train, cv=kfold, scoring=scoring)



    # Cross validation result


    # Result on testSet
    # calculate ROC_AUC
    auc_score = roc_auc_score(y_test, model.predict(testSet))
    # Generate the ROC curve
    fpr, tpr, threshold = roc_curve(y_test, model.predict(testSet))
    msg = "%s: %s %f (%f) %s: %f " % (name, "ROC_AUC on CV", 1.111111111111,0.1111,#cv_results.mean(), cv_results.std(),
                                      "; ROC_AUC on testSet", auc_score)
    print(msg)

    # Plot the ROC curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data')

    plt.plot(fpr, tpr, label="djLGB")

    plt.title('ROC Curves')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend(loc='lower left')

    # Let matplotlib improve the layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    #plt.show()

    # import os
    # default_path = os.getcwd()
    # print(default_path)  # Prints the working directory
    # #os.chdir(path)

    # To save the plot to an image file, use savefig()
    plt.savefig('../../reports/figures/plotdjLGB.png')

    # Open the image file with the default image viewer
    import subprocess
    subprocess.Popen('plotdjLGB.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    # from io import BytesIO
    # img_stream = BytesIO()
    # plt.savefig(img_stream, fmt='png')
    # img_bytes = img_stream.getvalue()
    # print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()