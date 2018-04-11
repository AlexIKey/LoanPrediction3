def select_features_lr(data, scoring='accuracy'):
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
    from sklearn.linear_model import LinearRegression
    from src.models.train_model import get_features_and_labels
    X, _,y, _ = get_features_and_labels(data)
    print(type(y))
    lr = LinearRegression()
    efs = EFS(lr,
          min_features=1,
          max_features=4,
          scoring=scoring,
          print_progress=True,
          cv=0)
    print('Start EFS procedure')
    efs.fit(X, y)

    print('Best MSE score: %.2f' % efs.best_score_ * (-1))
    print('Best subset:', efs.best_idx_)
    return efs.best_idx_


def select_features_efs(data, clf, scoring='accuracy', seed=1234):
    '''
    :param data:
    :param clf:
    :param scoring:
    :param seed:
    :return:
    '''
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

    efs1 = EFS(clf,
               min_features=5,
               max_features=10,
               scoring=scoring,
               print_progress=True,
               cv=0)
    efs1 = efs1.fit(X_train, y_train)
    print('Selected features:', efs1.best_idx_)
    return efs1.best_idx_

def select_features_rfe(data, clf, scoring='accuracy', seed=1234):
    from sklearn.feature_selection import RFE
    select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)
    select.fit(X_train,
               y_train)
    # визуализируем отобранные признаки :
    mask = select.get_support()
    plt.matshow(mask.reshape(1, -1), cmap='gray_r') plt.xlabel("Индекс примера")

    X_train_rfe= select.transform(X_train)
    X_test_rfe= select.transform(X_test)
    score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
    print("Правильность на тестовом наборе: {:.3f}".format(score))






if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from src.models.train_model import get_features_and_labels
    data_tmp = pd.read_csv('../../data/processed/train_ready.csv', index_col=0)
    seed = 12345

    X_train, X_test, y_train, y_test = get_features_and_labels(data_tmp, seed)
    clf = LogisticRegression()
    pos = list(select_features_efs(data_tmp, clf, scoring='accuracy'))
    features = data_tmp.columns[pos]
    print(features)
