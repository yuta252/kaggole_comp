import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

from load_data import load_train_data, load_test_data


logger = getLogger(__name__)
DIR = 'result_tmp/'


if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'feature_eng.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    train = load_train_data()
    test = load_test_data()
    logger.info('Train shape: {}, Test shape: {}'.format(train.shape, test.shape))
    
    #Drop Id columns
    train_ID = train['Id']
    test_ID = test['Id']
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)

    #delete the outlier
    train = train.drop(train[(train['GrLivArea'] > 4000 )&(train['SalePrice'] < 300000)].index)
    ntrain = train.shape[0]
    ntest = test.shape[0]

    logger.info('Train and test shape after deleting outlier:Train:{}, Test:{}'.format(ntrain, ntest))

    #Log transform target
    train.SalePrice = np.log1p(train.SalePrice)
    y = train.SalePrice
    train.drop('SalePrice', axis=1, inplace=True)

    #Handle missing values
    all_df = pd.concat((train, test))

    logger.info('Shape of all_df: {}'.format(all_df.shape))

    # Alley : data description says NA means "no alley access"
    all_df.loc[:, "Alley"] = all_df.loc[:, "Alley"].fillna("None")
    # BedroomAbvGr : NA most likely means 0
    all_df.loc[:, "BedroomAbvGr"] = all_df.loc[:, "BedroomAbvGr"].fillna(0)
    # BsmtQual etc : data description says NA for basement features is "no basement"
    all_df.loc[:, "BsmtQual"] = all_df.loc[:, "BsmtQual"].fillna("No")
    all_df.loc[:, "BsmtCond"] = all_df.loc[:, "BsmtCond"].fillna("No")
    all_df.loc[:, "BsmtExposure"] = all_df.loc[:, "BsmtExposure"].fillna("No")
    all_df.loc[:, "BsmtFinType1"] = all_df.loc[:, "BsmtFinType1"].fillna("No")
    all_df.loc[:, "BsmtFinType2"] = all_df.loc[:, "BsmtFinType2"].fillna("No")
    all_df.loc[:, "BsmtFullBath"] = all_df.loc[:, "BsmtFullBath"].fillna(0)
    all_df.loc[:, "BsmtHalfBath"] = all_df.loc[:, "BsmtHalfBath"].fillna(0)
    all_df.loc[:, "BsmtUnfSF"] = all_df.loc[:, "BsmtUnfSF"].fillna(0)
    # CentralAir : NA most likely means No
    all_df.loc[:, "CentralAir"] = all_df.loc[:, "CentralAir"].fillna("N")
    # Condition : NA most likely means Normal
    all_df.loc[:, "Condition1"] = all_df.loc[:, "Condition1"].fillna("Norm")
    all_df.loc[:, "Condition2"] = all_df.loc[:, "Condition2"].fillna("Norm")
    # EnclosedPorch : NA most likely means no enclosed porch
    all_df.loc[:, "EnclosedPorch"] = all_df.loc[:, "EnclosedPorch"].fillna(0)
    # External stuff : NA most likely means average
    all_df.loc[:, "ExterCond"] = all_df.loc[:, "ExterCond"].fillna("TA")
    all_df.loc[:, "ExterQual"] = all_df.loc[:, "ExterQual"].fillna("TA")
    # Fence : data description says NA means "no fence"
    all_df.loc[:, "Fence"] = all_df.loc[:, "Fence"].fillna("No")
    # FireplaceQu : data description says NA means "no fireplace"
    all_df.loc[:, "FireplaceQu"] = all_df.loc[:, "FireplaceQu"].fillna("No")
    all_df.loc[:, "Fireplaces"] = all_df.loc[:, "Fireplaces"].fillna(0)
    # Functional : data description says NA means typical
    all_df.loc[:, "Functional"] = all_df.loc[:, "Functional"].fillna("Typ")
    # GarageType etc : data description says NA for garage features is "no garage"
    all_df.loc[:, "GarageType"] = all_df.loc[:, "GarageType"].fillna("No")
    all_df.loc[:, "GarageFinish"] = all_df.loc[:, "GarageFinish"].fillna("No")
    all_df.loc[:, "GarageQual"] = all_df.loc[:, "GarageQual"].fillna("No")
    all_df.loc[:, "GarageCond"] = all_df.loc[:, "GarageCond"].fillna("No")
    all_df.loc[:, "GarageArea"] = all_df.loc[:, "GarageArea"].fillna(0)
    all_df.loc[:, "GarageCars"] = all_df.loc[:, "GarageCars"].fillna(0)
    # HalfBath : NA most likely means no half baths above grade
    all_df.loc[:, "HalfBath"] = all_df.loc[:, "HalfBath"].fillna(0)
    # HeatingQC : NA most likely means typical
    all_df.loc[:, "HeatingQC"] = all_df.loc[:, "HeatingQC"].fillna("TA")
    # KitchenAbvGr : NA most likely means 0
    all_df.loc[:, "KitchenAbvGr"] = all_df.loc[:, "KitchenAbvGr"].fillna(0)
    # KitchenQual : NA most likely means typical
    all_df.loc[:, "KitchenQual"] = all_df.loc[:, "KitchenQual"].fillna("TA")
    # LotFrontage : NA most likely means no lot frontage
    all_df.loc[:, "LotFrontage"] = all_df.loc[:, "LotFrontage"].fillna(0)
    # LotShape : NA most likely means regular
    all_df.loc[:, "LotShape"] = all_df.loc[:, "LotShape"].fillna("Reg")
    # MasVnrType : NA most likely means no veneer
    all_df.loc[:, "MasVnrType"] = all_df.loc[:, "MasVnrType"].fillna("None")
    all_df.loc[:, "MasVnrArea"] = all_df.loc[:, "MasVnrArea"].fillna(0)
    # MiscFeature : data description says NA means "no misc feature"
    all_df.loc[:, "MiscFeature"] = all_df.loc[:, "MiscFeature"].fillna("No")
    all_df.loc[:, "MiscVal"] = all_df.loc[:, "MiscVal"].fillna(0)
    # OpenPorchSF : NA most likely means no open porch
    all_df.loc[:, "OpenPorchSF"] = all_df.loc[:, "OpenPorchSF"].fillna(0)
    # PavedDrive : NA most likely means not paved
    all_df.loc[:, "PavedDrive"] = all_df.loc[:, "PavedDrive"].fillna("N")
    # PoolQC : data description says NA means "no pool"
    all_df.loc[:, "PoolQC"] = all_df.loc[:, "PoolQC"].fillna("No")
    all_df.loc[:, "PoolArea"] = all_df.loc[:, "PoolArea"].fillna(0)
    # SaleCondition : NA most likely means normal sale
    all_df.loc[:, "SaleCondition"] = all_df.loc[:, "SaleCondition"].fillna("Normal")
    # ScreenPorch : NA most likely means no screen porch
    all_df.loc[:, "ScreenPorch"] = all_df.loc[:, "ScreenPorch"].fillna(0)
    # TotRmsAbvGrd : NA most likely means 0
    all_df.loc[:, "TotRmsAbvGrd"] = all_df.loc[:, "TotRmsAbvGrd"].fillna(0)
    # Utilities : NA most likely means all public utilities
    all_df.loc[:, "Utilities"] = all_df.loc[:, "Utilities"].fillna("AllPub")
    # WoodDeckSF : NA most likely means no wood deck
    all_df.loc[:, "WoodDeckSF"] = all_df.loc[:, "WoodDeckSF"].fillna(0)    

    # Some numerical features are actually really categories
    all_df = all_df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                           "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                          })

    # Encode some categorical features as ordered numbers
    all_df = all_df.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}})


    # Create new features
    # 1* Simplifications of existing features
    all_df["SimplOverallQual"] = all_df.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                           4 : 2, 5 : 2, 6 : 2, # average
                                                           7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                          })
    all_df["SimplOverallCond"] = all_df.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                           4 : 2, 5 : 2, 6 : 2, # average
                                                           7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                          })
    all_df["SimplPoolQC"] = all_df.PoolQC.replace({1 : 1, 2 : 1, # average
                                                 3 : 2, 4 : 2 # good
                                                })
    all_df["SimplGarageCond"] = all_df.GarageCond.replace({1 : 1, # bad
                                                         2 : 1, 3 : 1, # average
                                                         4 : 2, 5 : 2 # good
                                                        })
    all_df["SimplGarageQual"] = all_df.GarageQual.replace({1 : 1, # bad
                                                         2 : 1, 3 : 1, # average
                                                         4 : 2, 5 : 2 # good
                                                        })
    all_df["SimplFireplaceQu"] = all_df.FireplaceQu.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    all_df["SimplFireplaceQu"] = all_df.FireplaceQu.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    all_df["SimplFunctional"] = all_df.Functional.replace({1 : 1, 2 : 1, # bad
                                                         3 : 2, 4 : 2, # major
                                                         5 : 3, 6 : 3, 7 : 3, # minor
                                                         8 : 4 # typical
                                                        })
    all_df["SimplKitchenQual"] = all_df.KitchenQual.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    all_df["SimplHeatingQC"] = all_df.HeatingQC.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    all_df["SimplBsmtFinType1"] = all_df.BsmtFinType1.replace({1 : 1, # unfinished
                                                             2 : 1, 3 : 1, # rec room
                                                             4 : 2, 5 : 2, 6 : 2 # living quarters
                                                            })
    all_df["SimplBsmtFinType2"] = all_df.BsmtFinType2.replace({1 : 1, # unfinished
                                                             2 : 1, 3 : 1, # rec room
                                                             4 : 2, 5 : 2, 6 : 2 # living quarters
                                                            })
    all_df["SimplBsmtCond"] = all_df.BsmtCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
    all_df["SimplBsmtQual"] = all_df.BsmtQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
    all_df["SimplExterCond"] = all_df.ExterCond.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    all_df["SimplExterQual"] = all_df.ExterQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })

    # 2* Combinations of existing features
    # Overall quality of the house
    all_df["OverallGrade"] = all_df["OverallQual"] * all_df["OverallCond"]
    # Overall quality of the garage
    all_df["GarageGrade"] = all_df["GarageQual"] * all_df["GarageCond"]
    # Overall quality of the exterior
    all_df["ExterGrade"] = all_df["ExterQual"] * all_df["ExterCond"]
    # Overall kitchen score
    all_df["KitchenScore"] = all_df["KitchenAbvGr"] * all_df["KitchenQual"]
    # Overall fireplace score
    all_df["FireplaceScore"] = all_df["Fireplaces"] * all_df["FireplaceQu"]
    # Overall garage score
    all_df["GarageScore"] = all_df["GarageArea"] * all_df["GarageQual"]
    # Overall pool score
    all_df["PoolScore"] = all_df["PoolArea"] * all_df["PoolQC"]
    # Simplified overall quality of the house
    all_df["SimplOverallGrade"] = all_df["SimplOverallQual"] * all_df["SimplOverallCond"]
    # Simplified overall quality of the exterior
    all_df["SimplExterGrade"] = all_df["SimplExterQual"] * all_df["SimplExterCond"]
    # Simplified overall pool score
    all_df["SimplPoolScore"] = all_df["PoolArea"] * all_df["SimplPoolQC"]
    # Simplified overall garage score
    all_df["SimplGarageScore"] = all_df["GarageArea"] * all_df["SimplGarageQual"]
    # Simplified overall fireplace score
    all_df["SimplFireplaceScore"] = all_df["Fireplaces"] * all_df["SimplFireplaceQu"]
    # Simplified overall kitchen score
    all_df["SimplKitchenScore"] = all_df["KitchenAbvGr"] * all_df["SimplKitchenQual"]
    # Total number of bathrooms
    all_df["TotalBath"] = all_df["BsmtFullBath"] + (0.5 * all_df["BsmtHalfBath"]) + \
    all_df["FullBath"] + (0.5 * all_df["HalfBath"])
    # Total SF for house (incl. basement)
    all_df["AllSF"] = all_df["GrLivArea"] + all_df["TotalBsmtSF"]
    # Total SF for 1st + 2nd floors
    all_df["AllFlrsSF"] = all_df["1stFlrSF"] + all_df["2ndFlrSF"]
    # Total SF for porch
    all_df["AllPorchSF"] = all_df["OpenPorchSF"] + all_df["EnclosedPorch"] + \
    all_df["3SsnPorch"] + all_df["ScreenPorch"]
    # Has masonry veneer or not
    all_df["HasMasVnr"] = all_df.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                                   "Stone" : 1, "None" : 0})
    # House completed before sale or not
    all_df["BoughtOffPlan"] = all_df.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                          "Family" : 0, "Normal" : 0, "Partial" : 1})

    # Create new features
    # 3* Polynomials on the top 10 existing features
    all_df["OverallQual-s2"] = all_df["OverallQual"] ** 2
    all_df["OverallQual-s3"] = all_df["OverallQual"] ** 3
    all_df["OverallQual-Sq"] = np.sqrt(all_df["OverallQual"])
    all_df["AllSF-2"] = all_df["AllSF"] ** 2
    all_df["AllSF-3"] = all_df["AllSF"] ** 3
    all_df["AllSF-Sq"] = np.sqrt(all_df["AllSF"])
    all_df["AllFlrsSF-2"] = all_df["AllFlrsSF"] ** 2
    all_df["AllFlrsSF-3"] = all_df["AllFlrsSF"] ** 3
    all_df["AllFlrsSF-Sq"] = np.sqrt(all_df["AllFlrsSF"])
    all_df["GrLivArea-2"] = all_df["GrLivArea"] ** 2
    all_df["GrLivArea-3"] = all_df["GrLivArea"] ** 3
    all_df["GrLivArea-Sq"] = np.sqrt(all_df["GrLivArea"])
    all_df["SimplOverallQual-s2"] = all_df["SimplOverallQual"] ** 2
    all_df["SimplOverallQual-s3"] = all_df["SimplOverallQual"] ** 3
    all_df["SimplOverallQual-Sq"] = np.sqrt(all_df["SimplOverallQual"])
    all_df["ExterQual-2"] = all_df["ExterQual"] ** 2
    all_df["ExterQual-3"] = all_df["ExterQual"] ** 3
    all_df["ExterQual-Sq"] = np.sqrt(all_df["ExterQual"])
    all_df["GarageCars-2"] = all_df["GarageCars"] ** 2
    all_df["GarageCars-3"] = all_df["GarageCars"] ** 3
    all_df["GarageCars-Sq"] = np.sqrt(all_df["GarageCars"])
    all_df["TotalBath-2"] = all_df["TotalBath"] ** 2
    all_df["TotalBath-3"] = all_df["TotalBath"] ** 3
    all_df["TotalBath-Sq"] = np.sqrt(all_df["TotalBath"])
    all_df["KitchenQual-2"] = all_df["KitchenQual"] ** 2
    all_df["KitchenQual-3"] = all_df["KitchenQual"] ** 3
    all_df["KitchenQual-Sq"] = np.sqrt(all_df["KitchenQual"])
    all_df["GarageScore-2"] = all_df["GarageScore"] ** 2
    all_df["GarageScore-3"] = all_df["GarageScore"] ** 3
    all_df["GarageScore-Sq"] = np.sqrt(all_df["GarageScore"])

    # Differentiate numerical features (minus the target) and categorical features
    categorical_features = all_df.select_dtypes(include = ["object"]).columns
    numerical_features = all_df.select_dtypes(exclude = ["object"]).columns

    logger.info('Numerical_features:{}'.format(len(numerical_features)))
    logger.info('Categorical_features:{}'.format(len(categorical_features)))
    
    all_df_num = all_df[numerical_features]
    all_df_cat = all_df[categorical_features]

    logger.info('NA for numerical features in all_df:{}'.format(all_df_num.isnull().values.sum()))
    all_df_num = all_df_num.fillna(all_df_num[:ntrain].median())
    logger.info('Remaining NA for numerical features in all_df: {}'.format(all_df_num.isnull().values.sum()))

    # Log transform of the skewed nemerical features
    skewness = all_df_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    
    logger.info('{} skewed numerical features to log transform'.format(skewness.shape[0]))
    
    skewed_features = skewness.index
    all_df_num[skewed_features] = np.log1p(all_df_num[skewed_features])

    # Create dummy features for categorical values via one hot encoding
    all_df_cat = pd.get_dummies(all_df_cat)
    logger.info('Remaining NA for categorical features in all_df :{}'.format(all_df_cat.isnull().values.sum()))    

    # Standardize numerical values
    stdSc = StandardScaler()
    all_df_num.iloc[:ntrain, :] = stdSc.fit_transform(all_df_num.iloc[:ntrain, :])
    all_df_num.iloc[ntrain:, :] = stdSc.transform(all_df_num.iloc[ntrain:, :])

    logger.info('After standardized train:{}'.format(all_df_num.iloc[:ntrain,:].head(2)))
    logger.info('After standardized test:{}'.format(all_df_num.iloc[ntrain:,:].head(2)))



    # Join categorical and numerical features
    all_df = pd.concat([all_df_num, all_df_cat], axis=1)
    logger.info('New number of features:{}'.format(all_df.shape[1]))

    # Separate train and test data
    train = all_df[:ntrain]
    test = all_df[ntrain:]

    train = pd.concat([train, y], axis=1)
    test = pd.concat([test_ID, test], axis=1)

    logger.info('After preprocessing data size: Train:{}, Test{}'.format(train.shape, test.shape))
    # Save training data to csv file
    train.to_csv(DIR + 'train_preprocess.csv', index=False)
    test.to_csv(DIR + 'test_preprocess.csv', index=False)
    logger.info('end')


