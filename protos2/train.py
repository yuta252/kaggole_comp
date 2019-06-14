import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold,cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

from load_data import read_csv 


logger = getLogger(__name__)
DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'


# Define Cross validation evaluation
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1))
    return (rmse)

# Define evaluatino function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# Define Stacking model
class StackingAverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # we again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out of fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # now train the cloned metamodel using the out of fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta features for the final predictions which is done by the  metamodel
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


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
   

    # Cross validation strategy
    n_folds = 5

    # Lasso Regression
    lasso = Lasso(alpha=0.0005, random_state=1)
    #score = rmsle_cv(lasso)

    #logger.info('Lasso score:{:.4f}({:.4f})'.format(score.mean(), score.std()))

    # Kernel Ridge Regression
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    #score = rmsle_cv(KRR)

    #logger.info('KernelRidge score:{:.4f}({:.4f})'.format(score.mean(), score.std()))

    # Elastic Net regression
    ENet = ElasticNet(alpha=0.0005, l1_ratio=.9,random_state=3)
    #score = rmsle_cv(ENet)

    #logger.info('ElasticNet score:{:.4f}({:.4f})'.format(score.mean(), score.std()))
    
    # Gradient Boosting Regression
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15,
                                  min_samples_split=10, loss='huber', random_state=5)
    #score = rmsle_cv(GBoost)

    #logger.info('Gradient Boosting score:{:.4f}({:.4f})'.format(score.mean(), score.std()))

    # XGBoost
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817,
                            n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7, nthread=-1)
    #score = rmsle_cv(model_xgb)

    #logger.info('XGBoost score:{:.4f}({:.4f})'.format(score.mean(), score.std()))

    # LightGBM
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin=55,
                             bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9,
                            min_data_in_leaf=6,min_sum_hessian_in_leaf=11)
    #score = rmsle_cv(model_xgb)

    #logger.info('LightGBM socre:{:.4f}({:.4f})'.format(score.mean(), score.std()))

    # Stack 3models as base model
    stacked_averaged_models = StackingAverageModels(base_models= (ENet, GBoost, KRR), meta_model=lasso)
    #score = rmsle_cv(stacked_averaged_models)
    #logger.info('Stacking Averaged models score: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

    # Stacked regressor
    stacked_averaged_models.fit(train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    logger.info('Stacking regressor score: {}'.format(rmsle(y_train, stacked_train_pred)))

    # XGBoost
    model_xgb.fit(train, y_train)
    xgb_train_pred = model_xgb.predict(train)
    xgb_pred = np.expm1(model_xgb.predict(test))
    logger.info('XGBoost regressor score: {}'.format(rmsle(y_train, xgb_train_pred)))

    # LightGBM
    model_lgb.fit(train, y_train)
    lgb_train_pred = model_lgb.predict(train)
    lgb_pred = np.expm1(model_lgb.predict(test.values))
    logger.info('LightGBM regressor score: {}'.format(rmsle(y_train, lgb_train_pred)))

    # Ensemble prediction
    ensemble = stacked_pred * 0.7 + xgb_pred * 0.15 + lgb_pred * 0.15

    # Submission
    sub = pd.DataFrame()
    sub['Id'] = test_id
    sub['SalePrice'] = ensemble
    sub.to_csv(DIR + 'submission.csv', index=False)
