from statistics import median, mode

def make_features(df):
    '''
    :param df: dataframe, which concatenate from train and test set
    :return: df
    '''
    print('Start ', make_features.__name__)
    # replace NA
    df = make_null_features(df)
    df.isnull().sum()

    # Make new variables
    df = make_calc_features(df)

    # Make log features
    df = make_logs(df, log_features)

    # Make string features (as numeric)
    df = make_str_features(df)

    # check NA
    df.apply(lambda x: sum(x.isnull()),axis=0)

    # Drop features
    df = drop_correlated_features(df)
    print('Finish ', make_features.__name__)

    return df



def if_else(condition, a, b):
    if condition:
        return a
    else:
        return b

def make_time_features(data):
    """
    make datetime features (but not in this case)
    """
    print('Start ', make_time_features.__name__)
    print('Finish ', make_time_features.__name__)

    return (data, tm)

def make_null_features(data):
    """
    replace missing value
    """
    print('Start ', make_null_features.__name__)
    # replace in Dependents NA -> '0'
    data['Dependents'] = data['Dependents'].fillna('0')

    # replace in Married NA -> "No"
    data['Married'] = data['Married'].fillna('No')

    # replace in Gender NA -> "Male"
    data['Gender'] = data['Gender'].fillna('Male')

    # replace in Self_Employed NA -> "Yes" if ApplicantIncome >0 else 'No'
    data.loc[data.ApplicantIncome > 0, 'Self_Employed'] = \
        data.loc[data.ApplicantIncome > 0, 'Self_Employed'].fillna('Yes')
    data.loc[data.ApplicantIncome <= 0, 'Self_Employed'] = \
        data.loc[data.ApplicantIncome <= 0, 'Self_Employed'].fillna('No')

    # replace in Loan_Amount NA -> median(Loan_Amount)
    data['LoanAmount'] = data['LoanAmount'].fillna(median(data['LoanAmount']))

    # replace in Loan_Amount_Term NA -> mode(Loan_Amount_Term)
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(mode(data['Loan_Amount_Term']))

    # replace in Credit_History NA -> "Yes"
    data['Credit_History'] = data['Credit_History'].fillna(1)
    print('Finish ', make_null_features.__name__)
    return data

def make_calc_features(data):
    """
    feature engineering
    """
    print('Start ', make_calc_features.__name__)
    # calculate FamilySize
    data['numDependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
    data['FamilySize'] = data.apply(lambda x: x.numDependents + 2 if x.CoapplicantIncome > 0 or x.Married == 'Yes' \
        else x.numDependents + 1, axis=1)
    del data['numDependents']

    # calculate TotalIncome
    data['TotalIncome'] = data.ApplicantIncome + data.CoapplicantIncome

    # calculate TotalIncomePerson
    data['TotalIncomePerson'] = data.TotalIncome / data.FamilySize

    # calculate DTI
    data['DTI'] = data.LoanAmount / data.TotalIncome

    # calculate DTI by person
    data['DTI_person'] = data.LoanAmount / (data.TotalIncome / data.FamilySize)

    # calculate LoanAmountMonth monthly
    data['LoanAmountMonth'] = data.LoanAmount / data.Loan_Amount_Term

    # calculate DTI_month
    data['DTI_month'] = data.LoanAmountMonth / data.TotalIncome

    # calculate DTI_month by person
    data['DTI_month_person'] = data.LoanAmountMonth / (data.TotalIncome / data.FamilySize)
    print('Finish ', make_calc_features.__name__)
    return data

log_features = ['LOG_ApplicantIncome',
                'LOG_CoapplicantIncome',
                'LOG_LoanAmount',
                'LOG_Loan_Amount_Term',
                'LOG_TotalIncome',
                'LOG_TotalIncomePerson',
                'LOG_LoanAmountMonth',
                'LOG_DTI',
                'LOG_DTI_person',
                'LOG_DTI_month',
                'LOG_DTI_month_person'
                ]

def make_logs(data, log_features):
    """
    log(features)
    """
    print('Start ', make_logs.__name__)
    for name in log_features:
        data[name] = np.log(data[name[4:]].abs() + 1)  # 'LOG_' +
    print('Finish ', make_logs.__name__)
    return data

#str_features =  ['Gender', 'Married','Dependents', 'Education', 'Self_Employed', 'Property_Area']

def make_str_features(data):
    """
    recode string features to numeric
    """
    print('Start ', make_str_features.__name__)
    data = pd.get_dummies(data)

    # data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    # data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
    # data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
    # data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    # data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
    # data['Property_Area'] = data['Property_Area'].map({'Urban': 1, 'Semiurban': 2, 'Rural': 3})
    #print("Total size of united DataFrame after make dummies variables: ", data.shape)
    print('Finish ', make_str_features.__name__)
    return data

def drop_correlated_features(df, corr_treshhold = 0.95):
    # Create correlation matrix
    print('Start ', drop_correlated_features.__name__)
    corr_matrix = df.corr().abs()
    #print(corr_matrix)
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than corr_treshhold
    to_drop = [column for column in upper.columns if any(upper[column] > corr_treshhold)]
    print("Columns were droped: ", to_drop)
    df.drop(columns = to_drop, axis=1)
    print('Finish ', drop_correlated_features.__name__)
    return df



if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # import matplotlib.pyplot as plt

    df_train = pd.read_csv('../../data/raw/train_u6lujuX_CVtuZ9i.csv', index_col=0)
    df_test = pd.read_csv('../../data/raw/test_Y3wMUE5_7gLdaTN.csv', index_col=0)
    print('Sizes', df_train.shape, df_test.shape)

    # recode and save outcome vector
    y = df_train['Loan_Status'].map({'N': 0, 'Y': 1})
    del df_train['Loan_Status']
    print('Length of outcome vector: ', len(y))

    # all in one dataframe
    df = pd.concat([df_train, df_test])
    print("Total size of united DataFrame:", df.shape)

    # replace NA
    df = make_null_features(df)
    print(df.isnull().sum())

    # Make new variables
    df = make_calc_features(df)

    # Make log features
    df = make_logs(df, log_features)

    # Make dummies features
    df = make_str_features(df)
    print("Total size of united DataFrame after make dummies variables: ", df.shape)


    # 'WOEization' of continuous variables
    # import rpy2
    # print(rpy2.__version__)
    # from rpy2.robjects.packages import importr
    #
    # # import R's "base" package
    # base = importr('base')
    #
    # # import R's "utils" package
    # utils = importr('utils')
    # ---------------------------
    # from src.features.woecalc import WoE
    # woe_def = WoE()
    # woe = WoE(7, 30, spec_values={0: '0', 1: '1'}, v_type='c')
    # # Transform x1
    # woe.fit(df_train['ApplicantIncome'],y)
    # # Transform x2 using x1 transformation rules
    # #woe.transform(df_train['ApplicantIncome'])
    # fig = woe.plot()
    # plt.show(fig)
    # # make monotonic transformation with decreasing relationship hypothesis
    # woe_monotonic = woe.force_monotonic(hypothesis=0)
    # fig = woe_monotonic.plot()
    # plt.show(fig)
    # N = 300
    # woe2 = woe.optimize(max_depth=5, min_samples_leaf=int(N / 3))
    # woe2 = woe.optimize(max_depth=3, scoring='r2')
    # fig2 = woe2.plot()
    # plt.show(fig2)
    ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # Drop features
    # drop correlated features
    df = drop_correlated_features(df)

    data_train = df[df.index.isin(df_train.index)]
    data_test = df[df.index.isin(df_test.index)]
    print(data_train.shape, data_test.shape)

    data_tmp = data_train.copy()
    data_tmp['y'] = y

    data_tmp.to_csv('../../data/processed/train_ready.csv', index=False)
    data_test.to_csv('../../data/processed/test_ready.csv', index=False)

