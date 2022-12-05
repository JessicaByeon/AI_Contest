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
        train_size=0.8, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,8,10,12], 'min_samples_leaf': [3,5,7,10]},          # 32번
    {'max_depth': [6,8,10,12], 'min_samples_leaf': [3,5,7,10], 'min_samples_split': [2,3,5,10]},    # 64번
    {'min_samples_leaf': [3,5,7,10], 'min_samples_split': [2,3,5,10], 'n_jobs': [-1,2,4]}           # 48번 = 총 144번
]



#2. 모델구성
from sklearn.svm import LinearSVC, SVC # 레거시한 리니어 모델 사용
from sklearn.linear_model import Perceptron, LogisticRegression # logistic regression 분류! (논리적회귀)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # 분류할 때 가지치기 형식으로 레이어 형식과 비슷하게 분류 
from sklearn.ensemble import RandomForestClassifier # decision tree 가 여러개 앙상블로 엮여있는데 랜덤하게 분류 / decision tree 보다 일반적으로 더 성능이 좋지만 반대인 경우도 있음
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) 


#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_) # 가장 좋은 추정치
print("최적의 파라미터  : ", model.best_params_)
print('best_score : ', model.best_score_) # 정확도
print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', accuracy_score(y_test, y_pred_best))

print('걸린시간 :', np.round(end-start, 4), "초")


y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_summit
submission.to_csv(path + 'sample_submission.csv', index=False)



# 3/ std, minmax scaler + 랜덤서치 + 랜덤포레스트 + StratifiedKFold
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=5, n_jobs=2)
# 최적의 파라미터  :  {'n_jobs': 2, 'min_samples_split': 5, 'min_samples_leaf': 3}
# best_score :  0.8612517408044564
# model.score :  0.8260869565217391
# accuracy_score :  0.8260869565217391
# 최적 튠 ACC : 0.8260869565217391
# 걸린시간 : 2.9536 초

# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=3)
# 최적의 파라미터  :  {'n_estimators': 100, 'min_samples_leaf': 3, 'max_depth': 12}
# best_score :  0.8599737855328909
# model.score :  0.8260869565217391
# accuracy_score :  0.8260869565217391
# 최적 튠 ACC : 0.8260869565217391
# 걸린시간 : 2.8919 초



# 2/ robust scaler + 랜덤포레스트 + stratified kfold
# ACC : [0.88746803 0.86700767 0.89258312 0.86189258 0.8797954 ] 
#  cross_val_score : 0.8777

# 1/ robust scaler + 랜덤포레스트
# 결과 acc : 0.8465473145780051
# RandomForestClassifier acc 스코어 :  0.8465473145780051






