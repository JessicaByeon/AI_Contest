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


###############################################################################################
# # 레이블 인코딩
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# train_set['TypeofContact'] = le.fit_transform(train_set['TypeofContact'])
# test_set['TypeofContact'] = le.fit_transform(test_set['TypeofContact'])
# train_set['Occupation'] = le.fit_transform(train_set['Occupation'])
# test_set['Occupation'] = le.fit_transform(test_set['Occupation'])
# train_set['Gender'] = le.fit_transform(train_set['Gender'])
# test_set['Gender'] = le.fit_transform(test_set['Gender'])
# train_set['ProductPitched'] = le.fit_transform(train_set['ProductPitched'])
# test_set['ProductPitched'] = le.fit_transform(test_set['ProductPitched'])
# train_set['MaritalStatus'] = le.fit_transform(train_set['MaritalStatus'])
# test_set['MaritalStatus'] = le.fit_transform(test_set['MaritalStatus'])
# train_set['Designation'] = le.fit_transform(train_set['Designation'])
# test_set['Designation'] = le.fit_transform(test_set['Designation'])


# # 결측치 처리
# # 1/ 중위값으로 처리
# train_set = train_set.fillna(train_set.median())
# test_set = test_set.fillna(test_set.median())
###############################################################################################

# 결측치 처리
# 1/ DurationOfPitch 결측치 0으로 채우기
train_set['DurationOfPitch'] = train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch'] = test_set['DurationOfPitch'].fillna(0)

# 2/ TypeofContact 결측치 "Unknown"으로 채우기
train_set['TypeofContact'] = train_set['TypeofContact'].fillna("Unknown")
test_set['TypeofContact'] = test_set['TypeofContact'].fillna("Unknown")

# 3/ Age 결측치 중위값으로 채우기 (최초 평균으로 채웠으나 중위값이 성능이 더 좋은 것으로 확인)
train_set['Age'] = train_set['Age'].fillna(train_set['Age'].median())
test_set['Age'] = test_set['Age'].fillna(test_set['Age'].median())

# 4/ Gender의 Fe male -> Female 로 변경
# print(train_set['Gender'].value_counts())
# train_set['Gender'] = train_set.replace({'Gender' : 'Fe Male'}, 'Female') # df = df.replace({'열 이름' : 기존 값}, 변경 값) 
# test_set['Gender'] = test_set.replace({'Gender' : 'Fe Male'}, 'Female') # ValueError: Columns must be same length as key [원인찾기]
train_set['Gender'] = train_set['Gender'].str.replace('Fe Male', 'Female')
test_set['Gender'] = train_set['Gender'].str.replace('Fe Male', 'Female')

# 5/ 직업 freelancer -> 최빈값(Salaried)으로 변경
# print(train_set['Occupation'].value_counts())
# train_set['Occupation'] = train_set['Occupation'].str.replace('Free Lancer', train_set['Occupation'].mode())
train_set['Occupation'] = train_set['Occupation'].str.replace('Free Lancer', 'Salaried')
test_set['Occupation'] = test_set['Occupation'].str.replace('Free Lancer', 'Salaried')

# # Age 범주화
# # train_set['AgeBand'] = pd.cut(train_set['Age'], 5)
# # 임의로 5개 그룹을 지정
# # print(train_set['AgeBand'])
# # [(17.957, 26.6] < (26.6, 35.2] < (35.2, 43.8] <
# # (43.8, 52.4] < (52.4, 61.0]]
# combine = [train_set,test_set]
# for dataset in combine:    
#     dataset.loc[ dataset['Age'] <= 26.6, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 26.6) & (dataset['Age'] <= 35.2), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 35.2) & (dataset['Age'] <= 43.8), 'Age'] = 2
#     dataset.loc[(dataset['Age'] > 43.8) & (dataset['Age'] <= 52.4), 'Age'] = 3
#     dataset.loc[ dataset['Age'] > 52.4, 'Age'] = 4
# # train_set = train_set.drop(['AgeBand'], axis=1)


# MonthlyIncome (Designation별 평균으로 변경)
# train_set['MonthlyIncome'] = train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'))
# test_set['MonthlyIncome'] = test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'))


# 나머지 결측치 중위값으로 채우기
# train_set['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome'
#           ] = train_set['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting'].fillna(train_set['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting'].median())
# test_set['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome'
#           ] = test_set['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting'].fillna(test_set['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting'].median())

median_cols = ['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome']
for col in median_cols:
    train_set[col] = train_set[col].fillna(train_set[col].median())
    test_set[col] = test_set[col].fillna(test_set[col].median())

# median_cols = ['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting']
# for col in median_cols:
#     train_set[col] = train_set[col].fillna(train_set[col].median())
#     test_set[col] = test_set[col].fillna(test_set[col].median())

print(train_set.isnull().sum())
print(test_set.isnull().sum())


# 문자열 데이터 레이블 인코딩
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


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.fit_transform(train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])

###############################################################################################
# 필요없는 컬럼 삭제
# train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)

# train_set = train_set.drop(['TypeofContact', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                             'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# test_set = test_set.drop(['TypeofContact', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                           'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9411764705882353

train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
                            'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
                          'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9372549019607843

# train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                             'Age', 'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                           'Age', 'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# # 결과 :  0.9176470588235294

# train_set = train_set.drop(['TypeofContact', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                             'Age', 'OwnCar', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# test_set = test_set.drop(['TypeofContact', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                           'Age', 'OwnCar', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9215686274509803

# train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                             'Age', 'OwnCar', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                           'Age', 'OwnCar', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9333333333333333


# Index(['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
#        'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
#        'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
#        'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
#        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
#        'ProdTaken'],
#       dtype='object')

# 'id' : '아이디'
# ,'Age' : '나이'
# ,'TypeofContact' : '탐색경로'
# ,'CityTier' : '도시등급'
# ,'DurationOfPitch' : '프리젠테이션기간'
# ,'Occupation' : '직업'
# ,'Gender' : '성별'
# ,'NumberOfPersonVisiting' : '여행인원'
# ,'NumberOfFollowups' : '후속조치수'
# ,'ProductPitched' : '제시상품'
# ,'PreferredPropertyStar' : '선호숙박등급'
# ,'MaritalStatus' : '결혼여부'
# ,'NumberOfTrips' : '연간여행횟수'
# ,'Passport' : '여권보유'
# ,'PitchSatisfactionScore' : '프레젠테이션만족도'
# ,'OwnCar' : '자동차보유'
# ,'NumberOfChildrenVisiting' : '미취학아동'
# ,'Designation' : '직급'
# ,'MonthlyIncome' : '월급여'
# ,'ProdTaken' : '신청여부'





# 모든 데이터 처리 완료 확인
# print(train_set.info())
###############################################################################################


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
        train_size=0.88, shuffle=True, random_state=666, stratify=y)

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


#2. 모델
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

'''
# baysian_params = {
#     'max_depth': (6, 16), # 범위
#     'min_child_weight': (1, 50),
#     'subsample': (0.5, 1),
#     'colsample_bytree': (0.5, 1),
#     'reg_lambda': (0.001, 10),
#     'reg_alpha': (0.01, 50),
# }

# baysian_params = {
#     'max_depth': (13, 16), # logloss 수정범위
#     'min_child_weight': (1, 5),
#     'subsample': (0.8, 1),
#     'colsample_bytree': (0.7, 1),
#     'reg_lambda': (0.001, 0.01),
#     'reg_alpha': (0.01, 0.1),
# }

baysian_params = {
    'max_depth': (6, 12), # error 수정범위
    'min_child_weight': (1, 5),
    'subsample': (0.8, 1),
    'colsample_bytree': (0.8, 1),
    'reg_lambda': (0.001, 0.1),
    'reg_alpha': (0.01, 0.1),
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
              eval_metric='error',
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

xgb_bo.maximize(init_points=5, n_iter=100)

print(xgb_bo.max)

##### 실습 #####
#1. 수정한 파라미터로 모델 만들어서 비교
#2. 수정한 파라미터를 이용해서 파라미터 재조정


# logloss
# 결과의 최대값 찾고
# {'target': 0.9319148936170213, 'params': {'colsample_bytree': 1.0, 'max_depth': 16.0, 'min_child_weight': 1.0, 'reg_alpha': 0.01, 'reg_lambda': 0.001, 'subsample': 1.0}} 
# 범위 수정하고 최대값 찾고
# {'target': 0.9446808510638298, 'params': {'colsample_bytree': 0.890591522677856, 'max_depth': 15.871586845348943, 'min_child_weight': 1.183714850334642, 'reg_alpha': 0.035292950805353836, 'reg_lambda': 0.005500806093034593, 'subsample': 0.8334895277425227}}

# error
# 결과의 최대값 찾고
# {'target': 0.9276595744680851, 'params': {'colsample_bytree': 1.0, 'max_depth': 8.658071873868758, 'min_child_weight': 1.0, 'reg_alpha': 0.01, 'reg_lambda': 0.001, 'subsample': 1.0}}
# 범위 수정하고 최대값 찾고
# {'target': 0.9404255319148936, 'params': {'colsample_bytree': 0.8528360237481907, 'max_depth': 10.85771027991774, 'min_child_weight': 1.0346758361743764, 'reg_alpha': 0.08624774344842114, 'reg_lambda': 0.01698848841429339, 'subsample': 0.9901176040652244}}


# 나온 실제 값을 xgb 모델에 넣어서 실행
'''
# eval_metric = error
# model = XGBClassifier(
#     n_estimators = 500, learning_rate = 0.02,
#     max_depth = int(round(10.85771027991774)), # 반올림해서 정수형으로 변환 (무조건 정수형)
#     min_child_weight = int(round(1.0346758361743764)),
#     subsample = max(min(0.9901176040652244, 1), 0), # 0~1 사이로 정규화
#     colsample_bytree = max(min(0.8528360237481907, 1), 0),
#     reg_lambda = max(0.01698848841429339, 0), # 무조건 양수만 받음
#     reg_alpha = max(0.08624774344842114, 0)
# )

# # eval_metric = logloss
model = XGBClassifier(
    n_estimators = 500, learning_rate = 0.02,
    max_depth = int(round(15.871586845348943)), # 반올림해서 정수형으로 변환 (무조건 정수형)
    min_child_weight = int(round(1.183714850334642)),
    subsample = max(min(0.8334895277425227, 1), 0), # 0~1 사이로 정규화
    colsample_bytree = max(min(0.890591522677856, 1), 0),
    reg_lambda = max(0.005500806093034593, 0), # 무조건 양수만 받음
    reg_alpha = max(0.035292950805353836, 0)
)

model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50
              )

y_predict = model.predict(x_test)
results = accuracy_score(y_test, y_predict)
print('결과 : ', results)


# submission 40_01
# logloss
# 결과 :  0.9446808510638298 --- 0.89428815

# submission 40_02
# error
# 결과 :  0.9404255319148936



y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_summit
submission.to_csv(path + 'sample_submission_40_01.csv', index=True)
