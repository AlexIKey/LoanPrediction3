# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd


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
    print('Length of outcome vector: ', len(y))

    df_test = pd.read_csv(input_filepath+'test_Y3wMUE5_7gLdaTN.csv', index_col=0)
    print('Sizes', df_train.shape, df_test.shape)

    # recode and save outcome vector
    y = df_train['Loan_Status'].map({'N': 0, 'Y': 1})
    del df_train['Loan_Status']

    # all in one dataframe
    df = pd.concat([df_train, df_test])
    df.shape

    import src.features.build_features make_features
    df = make_features(df)

    # Divide data on train and test again and save
    data_train = df[df.index.isin(df_train.index)]
    data_test = df[df.index.isin(df_test.index)]
    print(data_train.shape, data_test.shape)

    data_tmp = data_train.copy()
    data_tmp['y'] = y

    data_tmp.to_csv(output_filepath + 'train_ready.csv', index=False)
    data_test.to_csv(output_filepath + 'test_ready.csv', index=False)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
