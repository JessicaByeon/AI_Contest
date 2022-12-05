import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

path = 'D:/study_data/_data/dacon_radar/'
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

print(train_set)
print(train_set.shape) # (39607, 70)

print(train_set.columns)
print(train_set.info())
print(train_set.describe())

print(test_set)
print(test_set.shape) # (39608, 56)

# 결측치 확인 / 각 컬럼당 null 갯수 확인 --- 없음
print(train_set.isnull().sum())
print(test_set.isnull().sum())


train_x = train_set.filter(regex='X') # Input : X Featrue : 56
train_y = train_set.filter(regex='Y') # Output : Y Feature : 14


x_train, x_test, y_train, y_test = train_test_split(
    train_x, train_y, shuffle=True, random_state=1234, train_size=0.8)
# print(train_x.shape, train_y.shape)  #(39607, 56) (39607, 14)     
# print(test_set.shape) # (39608, 56)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


parameters = {'n_estimators': [100],
              'learning_rate' : [0.1],
            #   'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #   'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100],
            #   'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],
            #   'subsample' :[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            #   'colsample_bytree' : [0,1],
            #   'colsample_bylevel' : [0,1],
            #   'colsample_bynode' : [0,1],
            #   'reg_alpha' : [0, 1, 10],
            #   'reg_lambda' : [0, 1, 10],
              }

#2. 모델
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                    random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

# #######################################################################################
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.multioutput import MultiOutputRegressor

# model = MultiOutputRegressor(LinearRegression()).fit(x_train, y_train)
# # mor = MultiOutputRegressor(Ridge(random_state=123)).fit(x_train, y_train)
# # mor = MultiOutputRegressor(SVR()).fit(x_train, y_train)

# # model = RandomizedSearchCV(mor, #parameters,
# #                            cv=kfold, n_jobs=-1)
# # model = GridSearchCV(mor, parameters, cv=kfold, n_jobs=-1)
# #######################################################################################


model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('결과 : ', results)

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)


# submission_07
# 결과 :  0.06681104561504111
# 최상의 매개변수 :  {'n_estimators': 100, 'learning_rate': 0.1}
# 최상의 점수 :  0.06357421561683144 



y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (39608, 14)


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'sample_submission_07.csv', index=False)
