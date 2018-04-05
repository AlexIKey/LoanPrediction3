# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df_train = pd.read_csv(input_filepath+'train_u6lujuX_CVtuZ9i.csv', index_col=0)
    # recode and save outcome vector
    y = df_train['Loan_Status'].map({'N': 0, 'Y': 1})
    del df_train['Loan_Status']
    print('Length of outcome vector: ', len(y))

    df_test = pd.read_csv(input_filepath+'test_Y3wMUE5_7gLdaTN.csv', index_col=0)
    print('Sizes', df_train.shape, df_test.shape)

    # all in one dataframe
    df = pd.concat([df_train, df_test])
    df.shape

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

    # Divide data on train and test again and save
    data_train = df[df.index.isin(df_train.index)]
    data_test = df[df.index.isin(df_test.index)]
    print(data_train.shape, data_test.shape)

    data_tmp = data_train.copy()
    data_tmp['y'] = y

    data_tmp.to_csv(output_filepath+'train_ready.csv', index=False)
    data_test.to_csv(output_filepath+'test_ready.csv', index=False)



from statistics import median, mode

def if_else(condition, a, b):
    if condition:
        return a
    else:
        return b

def make_time_features(data):
    """
    make datetime features (but not in this case)
    """
    return (data, tm)

def make_null_features(data):
    """
    replace missing value
    """
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

    return data

def make_calc_features(data):
    """
    feature engineering
    """
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
    for name in log_features:
        data[name] = np.log(data[name[4:]].abs() + 1)  # 'LOG_' +

    return data

def make_str_features(data):
    """
    recode string features to numeric
    """
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
    data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
    data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
    data['Property_Area'] = data['Property_Area'].map({'Urban': 1, 'Semiurban': 2, 'Rural': 3})

    return data

def drop_correlated_features(df, corr_treshhold = 0.95):
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    #print(corr_matrix)
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than corr_treshhold
    to_drop = [column for column in upper.columns if any(upper[column] > corr_treshhold)]
    print("Columns were droped: ", to_drop)
    df.drop(columns = to_drop, axis=1)
    return df



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
