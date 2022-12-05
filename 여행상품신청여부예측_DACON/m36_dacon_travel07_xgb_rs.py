# 데이콘 문제풀기
# https://dacon.io/competitions/official/235959/

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook
from sklearn.metrics import accuracy_score

#1. 데이터

path = 'D:/study_data/_data/dacon_travel/'
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

print(train_set)
print(train_set.shape) # (1955, 19)
print(train_set.columns)
# Index(['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
#        'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
#        'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
#        'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
#        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
#        'ProdTaken'],
#       dtype='object')

print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
# <class 'pandas.core.frame.DataFrame'>        
# Int64Index: 1955 entries, 1 to 1955
# Data columns (total 19 columns):
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Age                       1861 non-null 
#   float64
#  1   TypeofContact             1945 non-null 
#   object
#  2   CityTier                  1955 non-null 
#   int64
#  3   DurationOfPitch           1853 non-null 
#   float64
#  4   Occupation                1955 non-null 
#   object
#  5   Gender                    1955 non-null 
#   object
#  6   NumberOfPersonVisiting    1955 non-null 
#   int64
#  7   NumberOfFollowups         1942 non-null 
#   float64
#  8   ProductPitched            1955 non-null 
#   object
#  9   PreferredPropertyStar     1945 non-null 
#   float64
#  10  MaritalStatus             1955 non-null 
#   object
#  11  NumberOfTrips             1898 non-null 
#   float64
#  12  Passport                  1955 non-null 
#   int64
#  13  PitchSatisfactionScore    1955 non-null 
#   int64
#  14  OwnCar                    1955 non-null 
#   int64
#  15  NumberOfChildrenVisiting  1928 non-null 
#   float64
#  16  Designation               1955 non-null 
#   object
#  17  MonthlyIncome             1855 non-null 
#   float64
#  18  ProdTaken                 1955 non-null 
#   int64
# dtypes: float64(7), int64(6), object(6)      
# memory usage: 305.5+ KB


print(train_set.describe())
#                Age  ...    ProdTaken
# count  1861.000000  ...  1955.000000
# mean     37.462117  ...     0.195908
# std       9.189948  ...     0.397000
# min      18.000000  ...     0.000000
# 25%      31.000000  ...     0.000000
# 50%      36.000000  ...     0.000000
# 75%      43.000000  ...     0.000000
# max      61.000000  ...     1.000000
# [8 rows x 13 columns]

print(test_set)
#        Age  ... MonthlyIncome
# id          ...
# 1     32.0  ...       19668.0
# 2     46.0  ...       20021.0
# 3     37.0  ...       21334.0
# 4     43.0  ...       22950.0
# 5     25.0  ...       21880.0
# ...    ...  ...           ...
# 2929  54.0  ...       32328.0
# 2930  33.0  ...       23733.0
# 2931  33.0  ...       23987.0
# 2932  26.0  ...       22102.0
# 2933  31.0  ...       22830.0
# [2933 rows x 18 columns]

print(test_set.shape) # (2933, 18)

# 결측치 확인
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능 -- age 177, cabin 687, embarked 2
# Age                          94
# TypeofContact                10
# CityTier                      0
# DurationOfPitch             102
# Occupation                    0
# Gender                        0
# NumberOfPersonVisiting        0
# NumberOfFollowups            13
# ProductPitched                0
# PreferredPropertyStar        10
# MaritalStatus                 0
# NumberOfTrips                57
# Passport                      0
# PitchSatisfactionScore        0
# OwnCar                        0
# NumberOfChildrenVisiting     27
# Designation                   0
# MonthlyIncome               100
# ProdTaken                     0
# dtype: int64

print(test_set.isnull().sum())
# Age                         132
# TypeofContact                15
# CityTier                      0
# DurationOfPitch             149
# Occupation                    0
# Gender                        0
# NumberOfPersonVisiting        0
# NumberOfFollowups            32
# ProductPitched                0
# PreferredPropertyStar        16
# MaritalStatus                 0
# NumberOfTrips                83
# Passport                      0
# PitchSatisfactionScore        0
# OwnCar                        0
# NumberOfChildrenVisiting     39
# Designation                   0
# MonthlyIncome               133
# dtype: int64



# Index(['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
#        'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',      
#        'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',   
#        'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
#        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',   
#        'ProdTaken'],
#       dtype='object')

# 레이블 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_set['TypeofContact'] = le.fit_transform(train_set['TypeofContact'])
test_set['TypeofContact'] = le.fit_transform(test_set['TypeofContact'])
train_set['Occupation'] = le.fit_transform(train_set['Occupation'])
test_set['Occupation'] = le.fit_transform(test_set['Occupation'])
train_set['Gender'] = le.fit_transform(train_set['Gender'])
test_set['Gender'] = le.fit_transform(test_set['Gender'])
train_set['ProductPitched'] = le.fit_transform(train_set['ProductPitched'])
test_set['ProductPitched'] = le.fit_transform(test_set['ProductPitched'])
train_set['PreferredPropertyStar'] = le.fit_transform(train_set['PreferredPropertyStar'])
test_set['PreferredPropertyStar'] = le.fit_transform(test_set['PreferredPropertyStar'])
train_set['MaritalStatus'] = le.fit_transform(train_set['MaritalStatus'])
test_set['MaritalStatus'] = le.fit_transform(test_set['MaritalStatus'])
train_set['Designation'] = le.fit_transform(train_set['Designation'])
test_set['Designation'] = le.fit_transform(test_set['Designation'])


# 결측치 처리
# 1/ 중위값으로 처리
train_set = train_set.fillna(train_set.median())
test_set = test_set.fillna(test_set.median())

print(train_set.isnull().sum())
print(test_set.isnull().sum())

    
x = train_set.drop(['ProdTaken'],axis=1) #axis는 컬럼 
print(x) #(1955, 18)
#        Age  TypeofContact  ...  Designation  MonthlyIncome
# id                         ...
# 1     28.0              0  ...    Executive        20384.0
# 2     34.0              1  ...      Manager        19599.0
# 3     45.0              0  ...      Manager        22295.0
# 4     29.0              0  ...    Executive        21274.0
# 5     42.0              1  ...      Manager        19907.0
# ...    ...            ...  ...          ...            ...
# 1951  28.0              1  ...    Executive        20723.0
# 1952  41.0              1  ...          AVP        31595.0
# 1953  38.0              0  ...    Executive        21651.0
# 1954  28.0              1  ...      Manager        22218.0
# 1955  22.0              0  ...    Executive        17853.0
# [1955 rows x 18 columns]

y = train_set['ProdTaken']
print(y.shape) #(1955,)

print(train_set.columns)
# Index(['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
#        'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',      
#        'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',   
#        'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
#        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',   
#        'ProdTaken'],
#       dtype='object')


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.85, shuffle=True, random_state=666, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=666, #stratify=y
                        )


# 'n_estimators': [100, 200, 300, 400, 500, 1000]} / 디폴트 100 / 1~inf / 정수 형태
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] / 디폴트 0.3 / 0~1 / eta 라고도 함
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] / 디폴트 6 / 0~inf / 정수 형태 / None은 무한대까지
# 통상적으로  depth를 낮게 잡을수록 성능이 좋고(4 정도), 깊게 잡을 수록 과적합 위험이 있다.
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] / 디폴트 0 / 0~inf / 정수 형태인지 찾아볼 것
# gamma 로스 값을 조각내는
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] / 디폴트 1 / 0~inf
# 'subsample' :[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha 라고도 함
# 'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda 라고도 함


parameters = {'n_estimators': [200],
              'learning_rate' : [0.1],
              'max_depth' : [7],
              'gamma' : [0],
              'min_child_weight' : [0.1],
              'subsample' :[1],
              'colsample_bytree' : [1],
              'colsample_bylevel' : [1],
              'colsample_bynode' : [1],
              'reg_alpha' : [0],
              'reg_lambda' : [0],
              }


#2. 모델
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                    random_state=666)


# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('결과 : ', results)

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

# 1/ submission 10
# 결과 :  0.8775510204081632

# 2/ submission 11
# 결과 :  0.8979591836734694

# 3/ submission 12 / 랜덤스테이트 123 -> 1234
# 결과 :  0.8945578231292517

# 결과 :  0.9013605442176871 / 랜덤스테이트 666 --- 0.8976982097


y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_summit
submission.to_csv(path + 'sample_submission_12.csv', index=True)


# 결과 :  0.8775510204081632
# 최상의 매개변수 :  {'n_estimators': 200}  
# 최상의 점수 :  0.8753826115271899

# 결과 :  0.891156462585034
# 최상의 매개변수 :  {'n_estimators': 200, 'learning_rate': 0.1}
# 최상의 점수 :  0.8759723579000687

# 결과 :  0.8945578231292517
# 최상의 매개변수 :  {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1}      
# 최상의 점수 :  0.8783856145301929

# 결과 :  0.8945578231292517
# 최상의 매개변수 :  {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1}      
# 최상의 점수 :  0.8783856145301929

# 결과 :  0.8945578231292517
# 최상의 매개변수 :  {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0}
# 최상의 점수 :  0.8783856145301929

# 결과 :  0.8979591836734694
# 최상의 매개변수 :  {'n_estimators': 200, 'min_child_weight': 0.1, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0}
# 최상의 점수 :  0.8838036832012734

# 결과 :  0.8979591836734694
# 최상의 매개변수 :  {'subsample': 1, 'n_estimators': 200, 'min_child_weight': 0.1, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0}
# 최상의 점수 :  0.8838036832012734

# 결과 :  0.8979591836734694
# 최상의 매개변수 :  {'subsample': 1, 'n_estimators': 200, 'min_child_weight': 0.1, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1}
# 최상의 점수 :  0.8838036832012734

# 결과 :  0.8979591836734694
# 최상의 매개변수 :  {'subsample': 1, 'n_estimators': 200, 'min_child_weight': 0.1, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1, 'colsample_bylevel': 1}
# 최상의 점수 :  0.8838036832012734

# 결과 :  0.8979591836734694
# 최상의 매개변수 :  {'subsample': 1, 'n_estimators': 200, 'min_child_weight': 0.1, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1, 'colsample_bynode': 1, 'colsample_bylevel': 1}
# 최상의 점수 :  0.8838036832012734

# 결과 :  0.8979591836734694
# 최상의 매개변수 :  {'subsample': 1, 'reg_alpha': 0, 'n_estimators': 200, 'min_child_weight': 0.1, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 
# 1, 'colsample_bynode': 1, 'colsample_bylevel': 1}
# 최상의 점수 :  0.8838036832012734

# 결과 :  0.8979591836734694
# 최상의 매개변수 :  {'subsample': 1, 'reg_lambda': 0, 'reg_alpha': 0, 'n_estimators': 200, 'min_child_weight': 0.1, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1, 'colsample_bynode': 1, 'colsample_bylevel': 1}
# 최상의 점수 :  0.8844042838018741