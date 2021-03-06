import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

def get_features_and_labels(frame, seed=None):
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    seed = int, RandomState instance or None, optional (default=None)
    !!!Use the last column as the target value!!!
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

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

def eval_classifiers(X_train, X_test, y_train, y_test, seed=None):
    """
        Run multiple times with different classifiers to get an idea of the
        relative performance of each configuration.

        Returns a sequence of tuples containing:
            (title, precision, recall)
        for each learner.
    """

    # Spot Check Algorithms

    from sklearn import model_selection
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier


    # We will calculate the P-R curve for each classifier
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    from sklearn.model_selection import KFold, StratifiedKFold



    scoring = 'roc_auc'
    models = []
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC()))
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('XGB', XGBClassifier()))
    models.append(('LGB', LGBMClassifier(verbose=-1)))


    # evaluate each model in turn
    details_cv = []
    results_cv = []
    results_test = []
    names = []

    for name, model in models:
        #kfold = model_selection.KFold(n_splits=10, random_state=seed)
        kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        # #############################################################################
        # Classification and ROC analysis

        # Run classifier with cross-validation and plot ROC curves
        tprs = []
        aucs = []
        each_cvs = []
        mean_fpr = np.linspace(0, 1, 100)


        i = 0
        for train, test in kfold.split(X_train, y_train):
            probas_ = model.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            each_cv = ('ROC fold %d (AUC = %0.2f)' % (i, roc_auc), fpr, tpr, roc_auc)
            each_cvs.append(each_cv)
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,
            #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i += 1


        # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        #          label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        # plt.plot(mean_fpr, mean_tpr, color='b',
        #          label=r'%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (name, mean_auc, std_auc),
        #          lw=2, alpha=.8)
        each_cv = (r'%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (name, mean_auc, std_auc), mean_fpr, mean_tpr, mean_auc)
        each_cvs.append(each_cv)
        details_cv.append(each_cvs)

        res_cv = tuple([r'%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (name, mean_auc, std_auc), mean_fpr, mean_tpr, mean_auc])
        results_cv.append(res_cv)

        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                  label=r'$\pm$ 1 std. dev.')
        #
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()
        # ================================================================================================

        #kfold = 10
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        # results.append(cv_results)
        # names.append(name)
        #
        # Result on testSet
        # calculate ROC_AUC
        #
        # Fit the classifier
        model.fit(X_train, y_train)
        auc_score = roc_auc_score(y_test, model.predict(X_test))
        # Generate the ROC curve
        fpr, tpr, threshold = roc_curve(y_test, model.predict(X_test))
        msg = "%s: %s %f (%f) %s: %f " % (name, "ROC_AUC on CV", cv_results.mean(), cv_results.std(),
                                           "; ROC_AUC on testSet", auc_score)
        print(msg)
        res_test = (name+' (AUC score={:.3f})'.format(auc_score), fpr, tpr, auc_score)
        results_test.append(res_test)
        #yield name+' (AUC score={:.3f})'.format(auc_score), fpr, tpr, auc_score
    # #####################################################################################
    # =====================================================================
    return details_cv, results_cv, results_test






def plot(title, results):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, fpr, tpr)

    All the elements in results will be plotted.
    '''

    #import matplotlib.pyplot as plt
    # Plot the ROC curves

    fig = plt.figure(figsize=(5, 5))
    fig.canvas.set_window_title('Classifying data')

    for label, fpr, tpr, _ in results:
        plt.plot(fpr, tpr, label=label)

    plt.title(title, color="blue")
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend(loc='lower right', fontsize=10)

    # Let matplotlib improve the layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('../../reports/figures/plotClassifiers.png')
    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plotClassifiers.png', shell=True)

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





if __name__ == "__main__":
    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt

    # -----------------------------------------------------------
    import os
    mingw_path = 'C:\\Program Files\\Git\\mingw64\\bin'
    os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
    # -----------------------------------------------------------
    seed = 1234567
    data_tmp = pd.read_csv('../../data/processed/train_ready.csv', index_col=0)

    trainSet, testSet, y_train, y_test = get_features_and_labels(data_tmp, seed=seed)

    # Evaluate multiple classifiers on the data
    print("Evaluating classifiers")
    details_cv, results_cv, results_test = list(eval_classifiers(trainSet, testSet, y_train, y_test, seed=seed))
    # indexing results
    import operator
    results_cv.sort(key=operator.itemgetter(3), reverse=True)
    results_test.sort(key=operator.itemgetter(3), reverse=True)
    # Display the results
    print("Plotting the results")

    #fig = plt.figure(figsize=(10, 25))
    fig, axis = plt.subplots(nrows=round(len(details_cv)/2), ncols=2, figsize=(10, 25))
    axis = axis.flatten()
    i = 0
    for detail_cv in details_cv:
        for label, fpr, tpr, _ in detail_cv:
            axis[i].plot(fpr, tpr,  lw=1, alpha=0.3, label=label)
            axis[i].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                         label='Luck', alpha=.8)
            axis[i].set
            axis[i].set_xlabel('fpr', fontsize=10)
            axis[i].set_ylabel('tpr', fontsize=10)
            axis[i].set_title(label[:label.find(' ')],  fontsize=10, color="blue")
            axis[i].legend(loc='lower right', fontsize=4)

        i += 1


    # Let matplotlib improve the layout
    fig.tight_layout()
    plt.show()



    plot('ROC Curves on CV Set', results_cv)
    plot('ROC Curves on Test Set', results_test)



