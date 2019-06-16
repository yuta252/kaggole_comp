"""
    Search the hyperparameters of lasso regression
"""
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from load_data import read_csv


logger = getLogger(__name__)
DIR = 'result_tmp/'


def rmse_cv(model):
    cv = KFold(n_folds, shuffle=True, random_state=123).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
    return (rmse)


if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'lasso_train.py.log', 'a')
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
   
    
    # Cross validation of Lasso regression
    logger.info('Lasso regression')
    lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1], max_iter = 50000, cv=5)  
    lasso.fit(train, y_train)
    alpha = lasso.alpha_
    logger.info("Best alpha :{}".format(alpha)) 
    print('Try to search again more precision around best alpha:{}'.format(alpha))
    lasso = LassoCV(alphas = [alpha * rate for rate in np.arange(0.6, 1.4, 0.05)], max_iter=50000, cv=5)
    lasso.fit(train, y_train)
    alpha = lasso.alpha_
    logger.info("Best alpha :{}".format(alpha)) 

    n_folds = 5
    logger.info('Lasso RMSE mean_score:{}'.format(rmse_cv(lasso).mean()))
    logger.info('Lasso RMSE std_socre:{}'.format(rmse_cv(lasso).std()))
    y_test_las = lasso.predict(test)
    
    logger.info('prediction of test :{}'.format(y_test_las.shape))

    with open(DIR + 'lasso.pkl', 'wb') as f:
        pickle.dump([lasso, y_test_las], f)


    # Ridge regression
    logger.info('Ridge regression')
    ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    ridge.fit(train, y_train)
    alpha = ridge.alpha_
    logger.info("Best alpha :{}".format(alpha)) 
    print('Try to search again more precision around best alpha:{}'.format(alpha))
    ridge = RidgeCV(alphas = [alpha * rate for rate in np.arange(0.6, 1.4, 0.05)], cv=5)
    ridge.fit(train, y_train)
    alpha = ridge.alpha_
    logger.info('Best alpha:{}'.format(alpha))
    
    n_folds = 5
    logger.info('Ridge RMSE mean_score:{}'.format(rmse_cv(ridge).mean()))
    logger.info('Ridge RMSE std_socre:{}'.format(rmse_cv(ridge).std()))
    y_test_ridge = ridge.predict(test)
    
    logger.info('prediction of test :{}'.format(y_test_ridge.shape))

    with open(DIR + 'ridge.pkl', 'wb') as f:
        pickle.dump([ridge, y_test_ridge], f)


    # Elastic Net
    elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                              alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                              max_iter = 50000, cv = 5)
    elasticNet.fit(train, y_train)
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    logger.info("Best alpha :{}, Best l1_ration:{}".format(alpha, ratio))

    logger.info("Try again for more precision with l1_ratio centered around {} ".format(ratio))
    elasticNet = ElasticNetCV(l1_ratio = [ratio * rate for rate in np.arange(0.85, 1.15, 0.05)],
                              alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                              max_iter = 50000, cv = 10)
    elasticNet.fit(train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    logger.info("Best alpha: {}, Best l1_ratio: {}".format(alpha, ratio))

    logger.info("Now try again for more precision on alpha, with l1_ratio fixed at {} and alpha centered around {}".format(ratio, alpha))
    elasticNet = ElasticNetCV(l1_ratio = ratio,
                              alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9,
                                        alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3,
                                        alpha * 1.35, alpha * 1.4],
                              max_iter = 50000, cv = 10)
    elasticNet.fit(train, y_train)
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    logger.info("Best alpha: {}, Best l1_ratio: {}".format(alpha, ratio))

    n_folds = 5
    logger.info('ElasticNet RMSE mean_score:{}'.format(rmse_cv(elasticNet).mean()))
    logger.info('ElasticNet RMSE std_socre:{}'.format(rmse_cv(elasticNet).std()))
    y_test_elas = elasticNet.predict(test)
    
    logger.info('prediction of test :{}'.format(y_test_elas.shape))

    with open(DIR + 'elasticnet.pkl', 'wb') as f:
        pickle.dump([elasticNet, y_test_elas], f)









