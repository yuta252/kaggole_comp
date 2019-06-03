import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LinearRegression

from load_data import load_train_data, load_test_data


logger = getLogger(__name__)
DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df = load_train_data()
    x_train = df.drop('SalePrice', axis=1)
    x_train_feature = x_train[['YearBuilt']].values
    y_train = df['SalePrice'].values
    
    # match test data clumns to train data columns
    use_cols = x_train.columns.values

    # logger.info('train columns: {}{}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end{}'.format(x_train.shape))

    clf = LinearRegression()
    clf.fit(x_train_feature, y_train)

    logger.info('train end')

    df = load_test_data()

    x_test = df[use_cols].sort_values('Id')
    x_test_feature = x_test[['YearBuilt']]

    logger.info('test data load end{}'.format(x_test.shape))
    pred_test = clf.predict(x_test_feature)

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('Id')
    df_submit['SalePrice'] = pred_test

    df_submit.to_csv(DIR + 'submit.csv', index=False)
    
    logger.info('end')


