import os
#Library in Data_Itergration class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

#Library in Data preprocessing class
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

#Library in Outlier Handling Class
#Univariate
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns
#Multivariate
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

#Library in Rescale class
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

#Library in Feature selection class
# Classification
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, chi2, RFE, RFECV, f_classif,mutual_info_regression,f_regression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import LinearSVC, SVC

# Regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.svm import LinearSVR, SVR

#Library in Model Selection class
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,recall_score,precision_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import SVC, NuSVC
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier
from xgboost              import XGBClassifier
from catboost             import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, max_error, mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDRegressor, HuberRegressor,TheilSenRegressor, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from imblearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

#Librabry in External Selection Class
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, max_error, mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDRegressor, HuberRegressor,TheilSenRegressor, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import SVC, NuSVC
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier
from xgboost              import XGBClassifier
from catboost             import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB

#Library in Data Sampling class
#!!pip install imbalanced-learn
from imblearn.under_sampling import TomekLinks,RandomUnderSampler,EditedNearestNeighbours,OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.under_sampling import  CondensedNearestNeighbour
from imblearn.over_sampling import ADASYN,RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from numpy import mean

#Lirbrary for deploying model
import pickle

