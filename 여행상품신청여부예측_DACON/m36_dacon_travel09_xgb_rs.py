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

# print(train_set)
# print(train_set.shape) # (1955, 19)
# print(train_set.columns)

# print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
# print(train_set.describe())

# print(test_set)
# print(test_set.shape) # (2933, 18)

# 결측치 확인
print(train_set.isnull().sum())
print(test_set.isnull().sum())


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

# 6/ MonthlyIncome (Designation별 평균으로 변경)
# train_set['MonthlyIncome'] = train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'))
# test_set['MonthlyIncome'] = test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'))


# 7/ 나머지 결측치 중위값으로 채우기
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


# 모든 데이터 처리 완료 확인
# print(train_set.info())


x = train_set.drop(['ProdTaken'],axis=1) #axis는 컬럼 
print(x) #(1955, 18)

y = train_set['ProdTaken']
print(y.shape) #(1955,)

print(train_set.columns)


# ########################################################################################
# # 아웃라이어 확인
# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
#     print("1사분위 :", quartile_1)
#     print("q2 :", q2)
#     print("3사분위 :", quartile_3)
#     iqr = quartile_3 - quartile_1
#     print("iqr :", iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((data_out>upper_bound) | # or
#                     (data_out<lower_bound))

# # 1사분위 : 1.0
# # q2 : 2.0
# # 3사분위 : 4.0
# # iqr : 3.0
# # 이상치의 위치 : (array([   0,    0,    0, ..., 1954, 1954, 1954], dtype=int64), 
# #                 array([ 0,  3, 17, ...,  0,  3, 17], dtype=int64))

# ### 슬라이싱
# outliers_loc = outliers(x)
# print("이상치의 위치 :", outliers_loc)

# # import matplotlib.pyplot as plt
# # plt.boxplot(outliers_loc)
# # plt.show()


# from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.25) # 25% 이상의 값을 이상치로 인식하도록 설정

# outliers.fit(x)
# results = outliers.predict(x)
# print(results) # [ 1  1  1 ... -1  1  1]
# ########################################################################################

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.94, shuffle=True, random_state=1234, stratify=y)


from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234, #stratify=y
                        )

# 1/ 기존 수정 전 파라미터
parameters = {'n_estimators': [200, 300],
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

# # 2/ 파라미터 재정의
# parameters = {'n_estimators': [400],
#               'learning_rate' : [0.2],
#               'max_depth' : [None],
#               'gamma' : [0],
#               'min_child_weight' : [1],
#               'subsample' : [1],
#               'colsample_bytree' : [1],
#               'colsample_bylevel' : [1],
#               'colsample_bynode' : [1],
#               'reg_alpha' : [0],
#               'reg_lambda' : [0],
#               }

# 3/ 아래의 매개변수 넣어서 돌려보기
# 최적의 매개변수 :  XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,     
#               colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
#               early_stopping_rounds=None, enable_categorical=False,
#               eval_metric=None, gamma=0, gpu_id=0, grow_policy='depthwise',
#               importance_type=None, interaction_constraints='',
#               learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
#               max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
#               missing=nan, monotone_constraints='()', n_estimators=400,
#               n_jobs=0, num_parallel_tree=1, predictor='gpu_predictor',
#               random_state=1234, reg_alpha=0, reg_lambda=1, ...)
# parameters = {'n_estimators': [400],
#               'learning_rate' : [0.300000012],
#               'max_depth' : [6],
#               'gamma' : [0],
#               'min_child_weight' : [1],
#               'subsample' :[1],
#               'colsample_bytree' : [1],
#               'colsample_bylevel' : [1],
#               'colsample_bynode' : [1],
#               'reg_alpha' : [0],
#               'reg_lambda' : [1],
#               }


#2. 모델
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                    random_state=1234)


# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
# model = HalvingGridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
# model = HalvingRandomSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('결과 : ', results)
# print("최적의 매개변수 : ", model.best_estimator_) # 가장 좋은 추정치
# print("최적의 파라미터  : ", model.best_params_)
# print('최상의 점수 : ', model.best_score_)

# 1/ submission 20 : 7번 파일에서 이상치 .25 .3 .2 제거 / gs, rs, hgs, hrs 넷 다 / train_size=0.87
# 결과 :  0.9058823529411765
# 최상의 점수 :  0.8729411764705883

# 2/ submission 21 : 'n_estimators': [200, 300] 로 300 추가
# 결과 :  0.9098039215686274

############### 2nd baseline 참고 ###############
# 1/ submission 30 : 기존 수정 전 파라미터
# 결과 :  0.9019607843137255

# 2/ submission 30_1 : 기존 수정 전 파라미터 / 변수조정 random state 1234
# 결과 :  0.9098039215686274
# 2/ submission 30_1 : 기존 수정 전 파라미터 / 변수조정 random state 12345
# 결과 :  0.9058823529411765
# 2/ submission 30_1 : 기존 수정 전 파라미터 / 변수조정 random state 666
# 결과 :  0.9019607843137255
# 2/ submission 30_1 : 기존 수정 전 파라미터 / 변수조정 random state 1234 / train_size=0.87 -> 0.84
# 결과 :  0.9137380191693291
# 2/ submission 30_2 : 기존 수정 전 파라미터 / 변수조정 random state 1234 / train_size=0.87 -> 0.93
# 결과 :  0.9051094890510949
# 2/ submission 30_2 : 기존 수정 전 파라미터 / 변수조정 random state 1234 / train_size=0.87 -> 0.94
# 결과 :  0.923728813559322 --- 0.8883205456
# 2/ submission 30_3 : 기존 수정 전 파라미터 / 변수조정 random state 1234 / train_size=0.87 -> 0.95
# 결과 :  0.9183673469387755

# 2/ submission 30_4 : 30_2 조건(0.94)에서 정규화 적용 위치 변경 --- 정규화 위치 변경하지말것...
# 결과 :  0.9183673469387755
# 2/ submission 30_4 : 30_2 조건(0.94)에서 age 결측치 평균 -> 중위값 변경
# 결과 :  0.9322033898305084 --- 0.8934356351
# 2/ submission 30_5 : 30_2 조건(0.94)에서 age 결측치 중위값 및 데이터 전처리 4~7 모두 적용
# 결과 :  0.9322033898305084 --- 0.8968456948
# 2/ submission 30_6 : 30_2 조건(0.94)에서 age 결측치 중위값 및 데이터 전처리 4~7 모두 적용(급여 직급별 평균)
# 결과 :  0.9322033898305084






# 3/ submission 31 : 2/ 파라미터 재정의 + train_size=0.94
# 결과 :  0.923728813559322 --- 0.8925831202	
# 3/ submission 31_1 : 3/ 파라미터 재정의 + train_size=0.94
# 결과 :  0.923728813559322



y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_summit
submission.to_csv(path + 'sample_submission_30_5.csv', index=True)
