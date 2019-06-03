import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from sklearn.metrics import mean_squared_error, r2_score

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

    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # parameter Search
    all_params = {'fit_intercept': [True, False]}
    max_score = 0
    max_prams = None

    for params in ParameterGrid(all_params):
        logger.info('params: {}'.format(params))

        list_mse_score = []
        list_r2_score = []
        # Cross validation
        for train_idx, valid_idx in cv.split(x_train_feature, y_train):
            trn_x = x_train_feature[train_idx]
            val_x = x_train_feature[valid_idx]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            clf = LinearRegression(**params)
            clf.fit(trn_x, trn_y)
        
            # validation
            pred = clf.predict(val_x)
            sc_mse = mean_squared_error(val_y, pred)
            sc_r2 = r2_score(val_y, pred)

            list_mse_score.append(sc_mse)
            list_r2_score.append(sc_r2)
            logger.debug('    MSE:{}, R2:{}'.format(sc_mse, sc_r2))
        sc_mse = np.mean(list_mse_score)
        sc_r2 = np.mean(list_r2_score)
        if sc_r2 > max_score:
            max_score = sc_r2
            max_params = params
        logger.info('MSE:{}, R2:{}'.format(sc_mse, sc_r2)) 
        logger.info('current max R2: {}, params: {}'.format(max_score, max_params))

    logger.info('maximum params: {}'.format(max_params))
    logger.info('maximum R2: {}'.format(max_score))

    clf = LinearRegression(**max_params)
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


