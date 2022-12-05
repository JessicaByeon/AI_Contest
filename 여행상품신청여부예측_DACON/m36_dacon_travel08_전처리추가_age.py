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

'''
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
'''

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

#2. 모델
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                    random_state=666)


# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
# model = HalvingGridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)
# model = HalvingRandomSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('결과 : ', results)
# print('최상의 점수 : ', model.best_score_)


# 1/ submission 20 : 7번 파일에서 이상치 .25 .3 .2 제거 / gs, rs, hgs, hrs 넷 다 / train_size=0.87
# 결과 :  0.9058823529411765
# 최상의 점수 :  0.8729411764705883

# 2/ submission 21 : 'n_estimators': [200, 300] 로 300 추가
# 결과 :  0.9098039215686274 --- 0.8994032396

#  21_3 : 'n_estimators': [200, 300] 로 300 추가 --- 여기에 나이 범주화
# 결과 :  0.9215686274509803 --- 0.8772378517


######################### 데이터전처리 재구성 #########################

# 3/ submission 23 : 
# drop(['NumberOfChildrenVisiting', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# drop(['NumberOfChildrenVisiting', 'NumberOfTrips', 'NumberOfFollowups', 'Age'], axis=1)
# drop(['NumberOfChildrenVisiting', 'NumberOfTrips', 'Age'], axis=1)
# drop(['CityTier', 'NumberOfTrips', 'Age'], axis=1)
# drop(['NumberOfTrips', 'Age'], axis=1)
# drop(['NumberOfFollowups', 'NumberOfTrips', 'Age'], axis=1) 
# drop(['Age'], axis=1) 결과 :  0.9058823529411765
# drop(['NumberOfChildrenVisiting', 'CityTier', 'NumberOfTrips', 'Age'], axis=1)
# drop(['NumberOfFollowups', 'CityTier', 'NumberOfTrips', 'Age'], axis=1)
# drop(['NumberOfFollowups', 'NumberOfTrips', 'Age'], axis=1)
# drop(['NumberOfChildrenVisiting', 'NumberOfFollowups'], axis=1)

# 결과 :  0.9215686274509803
# 결과 :  0.9254901960784314 --- 0.8772378517
# 결과 :  0.9176470588235294
# 결과 :  0.9176470588235294
# 결과 :  0.9215686274509803
# 결과 :  0.9215686274509803
# 결과 :  0.9058823529411765
# 결과 :  0.8941176470588236
# 결과 :  0.9215686274509803
# 결과 :  0.9215686274509803
# 결과 :  0.9176470588235294

# drop(['TypeofContact', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#       'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9411764705882353 -> 0.9011082694
# 급여 결측치 직급별 평균으로 채워넣은 후
# 결과 :  0.9411764705882353

# 3/ submission 23_01 : 
# drop(['PreferredPropertyStar', 'MaritalStatus', 'Designation'], axis=1)
# drop(['PreferredPropertyStar', 'MaritalStatus'], axis=1)
# drop(['PreferredPropertyStar', 'MaritalStatus', 'Age'], axis=1)
# drop(['PreferredPropertyStar', 'MaritalStatus', 'Designation', 'Age'], axis=1)
# drop(['PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Age'], axis=1)
# drop(['PreferredPropertyStar', 'MaritalStatus', 'NumberOfFollowups', 'Age'], axis=1)
# drop(['MaritalStatus', 'NumberOfFollowups', 'Age'], axis=1)
# drop(['PreferredPropertyStar', 'NumberOfFollowups', 'Age'], axis=1)
# drop(['MaritalStatus', 'NumberOfChildrenVisiting', 'NumberOfTrips', 'NumberOfFollowups', 'Age'], axis=1)

# 결과 :  0.8823529411764706
# 결과 :  0.8784313725490196
# 결과 :  0.9019607843137255
# 결과 :  0.8823529411764706
# 결과 :  0.8980392156862745
# 결과 :  0.8823529411764706
# 결과 :  0.8980392156862745
# 결과 :  0.8941176470588236
# 결과 :  0.9019607843137255

# drop(['TypeofContact', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#       'Age', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9176470588235294 (급여 직급별 평균 채워도 변동없음)
# drop(['TypeofContact', 'NumberOfChildrenVisiting',
#       'OwnCar','MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9254901960784314
# drop(['Age','OwnCar','MonthlyIncome'], axis=1)
# 결과 :  0.9254901960784314 --- 0.8772378517
# drop(['OwnCar','MonthlyIncome'], axis=1)
# 결과 :  0.9176470588235294


# 3/ submission 23_02 :
# drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#       'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9372549019607843 --- 0.9087809037

# 23_02 submission 0.87 -> 0.88 / 666
# 결과 :  0.948936170212766 --- 0.8994032396
# 0.88 / 1234
# 결과 :  0.9404255319148936
# 0.94
# 결과 :  0.940677966101695
# 0.85
# 결과 :  0.9455782312925171

# 3/ submission 23_03 :
# drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#       'Age', 'OwnCar', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
# 결과 :  0.9333333333333333 --- 0.8832054561



##### 급여를 직급별 평균으로 변경하면 결과 :  0.9254901960784314 -> 0.9215686274509803 로 낮아짐


y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_summit
submission.to_csv(path + 'sample_submission_23_02.csv', index=True)
