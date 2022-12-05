# 데이콘 문제풀기
# https://dacon.io/competitions/official/235959/

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
print(train_set.describe())

print(test_set)
print(test_set.shape) # (2933, 18)

# 결측치 확인
print(train_set.isnull().sum())
print(test_set.isnull().sum())

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

y = train_set['ProdTaken']
print(y.shape) #(1955,)

print(train_set.columns)


# 아웃라이어 확인
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 :", quartile_1)
    print("q2 :", q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | # or
                    (data_out<lower_bound))

# 1사분위 : 1.0
# q2 : 2.0
# 3사분위 : 4.0
# iqr : 3.0
# 이상치의 위치 : (array([   0,    0,    0, ..., 1954, 1954, 1954], dtype=int64), 
#                 array([ 0,  3, 17, ...,  0,  3, 17], dtype=int64))

### 슬라이싱
outliers_loc = outliers(x)
print("이상치의 위치 :", outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(outliers_loc)
# plt.show()


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.25) # 25% 이상의 값을 이상치로 인식하도록 설정

outliers.fit(x)
results = outliers.predict(x)
print(results) # [ 1  1  1 ... -1  1  1]


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.87, shuffle=True, random_state=666, stratify=y)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()

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


# parameters = {'n_estimators': [200, 300],
#               'learning_rate' : [0.1],
#               'max_depth' : [7],
#               'gamma' : [0],
#               'min_child_weight' : [0.1],
#               'subsample' :[1],
#               'colsample_bytree' : [1],
#               'colsample_bylevel' : [1],
#               'colsample_bynode' : [1],
#               'reg_alpha' : [0],
#               'reg_lambda' : [0],
#               }

# # CatBoost
# parameters = {'n_estimators': [200, 300],
#               'learning_rate' : [0.1],
#               'max_depth' : [7],
#               #'gamma' : [0],
#               #'min_child_weight' : [0.1],
#               'subsample' :[1],
#               #'colsample_bytree' : [1],
#               'colsample_bylevel' : [1],
#               #'colsample_bynode' : [1],
#               #'reg_alpha' : [0],
#               'reg_lambda' : [0],
#               }


#2. 모델
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')


baysian_params = {
    'max_depth': (6, 16), # 범위
    'min_child_weight': (1, 50),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.001, 10),
    'reg_alpha': (0.01, 50),
}


def xgb_hamsu(max_depth, min_child_weight, 
              subsample, colsample_bytree, reg_lambda, reg_alpha):
    params ={
        'n_estimators': 500, 'learning_rate' : 0.02,
        'max_depth': int(round(max_depth)), # 반올림해서 정수형으로 변환 (무조건 정수형)
        'min_child_weight': int(round(min_child_weight)),
        'subsample': max(min(subsample, 1), 0), # 0~1 사이로 정규화 / 1 이상이면 1로, 
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'reg_lambda': max(reg_lambda, 0), # 무조건 양수만 받음
        'reg_alpha': max(reg_alpha, 0),
    }
    
    # 함수 내부에 모델을 정의! 
    
    # **키워드받겠다(딕셔너리형태) 
    # *여러개의인자를받겠다 / 넣고싶은 인자를 1~n개 받아들이겠다.   
    model = XGBClassifier(**params) # baysian_params 를 받아서 params 에 따라 변환하여 넣어줌
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50
              )

    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results # 모델 실행한 것의 최대값, 어떤 파라미터를 넣었을 때 최대값이 나오는지를 찾자!

xgb_bo = BayesianOptimization(f=xgb_hamsu,
                              pbounds = baysian_params,
                              random_state=1234
                              )

xgb_bo.maximize(init_points=5, n_iter=20)

print(xgb_bo.max)

##### 실습 #####
#1. 수정한 파라미터로 모델 만들어서 비교
#2. 수정한 파라미터를 이용해서 파라미터 재조정


# 결과의 최대값 찾고
# {'target': 0.9215686274509803, 'params': {'colsample_bytree': 1.0, 'max_depth': 13.788199958489853, 'min_child_weight': 1.0, 'reg_alpha': 0.01, 'reg_lambda': 0.001, 'subsample': 1.0}}

# 범위 수정하고 최대값 찾고
# 수정하니 최대값이 줄어들어서 원래 값으로 아래 실행

# 나온 실제 값을 xgb 모델에 넣어서 실행

# eval_metric = error
model = XGBClassifier(
    n_estimators = 500, learning_rate = 0.02,
    max_depth = int(round(13.788199958489853)), # 반올림해서 정수형으로 변환 (무조건 정수형)
    min_child_weight = int(round(1.0)),
    subsample = max(min(1.0, 1), 0), # 0~1 사이로 정규화
    colsample_bytree = max(min(1.0, 1), 0),
    reg_lambda = max(0.001, 0), # 무조건 양수만 받음
    reg_alpha = max(0.01, 0)
)

# # eval_metric = logloss
# model = XGBClassifier(
#     n_estimators = 500, learning_rate = 0.02,
#     max_depth = int(round(16.0)), # 반올림해서 정수형으로 변환 (무조건 정수형)
#     min_child_weight = int(round(1.0)),
#     subsample = max(min(1.0, 1), 0), # 0~1 사이로 정규화
#     colsample_bytree = max(min(0.5, 1), 0),
#     reg_lambda = max(0.001, 0), # 무조건 양수만 받음
#     reg_alpha = max(0.01, 0)
# )

model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50
              )

y_predict = model.predict(x_test)
results = accuracy_score(y_test, y_predict)
print(results) 


# 1/ submission 20 : 7번 파일에서 이상치 .25 .3 .2 제거 / gs, rs, hgs, hrs 넷 다 / train_size=0.87
# 결과 :  0.9058823529411765 --- 0.8976982097

# 2/ submission 21 : 'n_estimators': [200, 300] 로 300 추가
# 결과 :  0.9098039215686274 --- 0.8994032396	

#############################################################

# 2/ submission 21_01 lgbm + rs
# 결과 :  0.9137254901960784 --- 0.8832054561

# 2/ submission 21_02 catboost + rs (가능 파라미터 확인할 것!)
# 결과 :  0.8980392156862745 --- 0.8917306053

#############################################################

# 3/ submission 25 xgb + bo / 25번 훈련, eval metric error
# 0.9215686274509803 --- 

# 3/ submission 25_01 xgb + bo / 25번 훈련, eval metric logloss
# 0.9137254901960784 --- 0.8883205456


y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_summit
submission.to_csv(path + 'sample_submission_25_01.csv', index=True)

