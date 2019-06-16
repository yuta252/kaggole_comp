"""
    Search the hyperparameters of Support vector machine regression
"""
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.svm import SVR 
from sklearn.model_selection import KFold, cross_val_score, train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error

from load_data import read_csv


logger = getLogger(__name__)
DIR = 'result_tmp/'


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'svm_linear.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    train = read_csv(DIR + 'train_preprocess.csv')
    test = read_csv(DIR + 'test_preprocess.csv')

    logger.info('Load train data shape:{}'.format(train.shape))

    y_train = train['SalePrice'].values
    train.drop('SalePrice', axis=1, inplace=True)

    test_id = test['Id']
    test.drop('Id', axis=1, inplace=True)


    logger.info('After loading')
    logger.info('train:{}, target: {}'.format(train.shape, y_train.shape))
    logger.info('test:{}, 1d: {}'.format(test.shape, test_id.shape))
   
    
    # Ridge kernel regression
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    all_params = {'C':[2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2],
                  'kernel':['linear'],
                  'epsilon':[2**-5, 2**-4, 2**-3, 2**-2, 2**-1],
                  'max_iter':[10000]}
    
    min_rmse = 0.50
    min_params = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params:{}'.format(params))
        
        list_rmse_score = []
        for train_idx, valid_idx in cv.split(train, y_train):
            trn_x = train.iloc[train_idx, :]
            val_x = train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            clf = SVR(**params)
            clf.fit(trn_x, trn_y)
            pred = clf.predict(val_x)
            sc_rmse = rmse(val_y, pred)
            
            list_rmse_score.append(sc_rmse)

        sc_rmse = np.mean(list_rmse_score)

        # evaluate rmse
        if sc_rmse < min_rmse:
            min_rmse = sc_rmse
            min_params = params
        logger.info('Current rmse score:{}, params:{}'.format(min_rmse, min_params))

    logger.info('Minimum params:{}'.format(min_params))
    logger.info('minimum rmse:{}'.format(min_rmse))

    # Train model with minimum paramas
    clf = SVR(**min_params)
    clf.fit(train, y_train)
    
    with open(DIR + 'svm_linear.pkl', 'wb') as f:
        pickle.dump(clf, f)

    logger.info('train end')    



