from time import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
import numpy as np 

#1. 데이터
path = 'D:/study_data/_data/dacon_travel/'
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

# print(train_set)
# print(train_set.shape) # (1955, 19)
# print(train_set.columns)
# Index(['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
#        'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
#        'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
#        'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
#        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
#        'ProdTaken'],
#       dtype='object')

# print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
# print(train_set.describe())

# print(test_set)
# print(test_set.shape) # (2933, 18)

# 결측치 확인
# print(train_set.isnull().sum())
# print(test_set.isnull().sum())

# 결측치 처리
# DurationOfPitch 결측치 0으로 채우기
train_set['DurationOfPitch'] = train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch'] = test_set['DurationOfPitch'].fillna(0)

# TypeofContact 결측치 "Self Enquiry"로 채우기
train_set['TypeofContact'] = train_set['TypeofContact'].fillna("Self Enquiry")
test_set['TypeofContact'] = test_set['TypeofContact'].fillna("Self Enquiry")

# Gender의 Fe male -> Female 로 변경
# print(train_set['Gender'].value_counts())
# train_set['Gender'] = train_set.replace({'Gender' : 'Fe Male'}, 'Female') # df = df.replace({'열 이름' : 기존 값}, 변경 값) 
# test_set['Gender'] = test_set.replace({'Gender' : 'Fe Male'}, 'Female') # ValueError: Columns must be same length as key [원인찾기]
train_set['Gender'] = train_set['Gender'].str.replace('Fe Male', 'Female')
test_set['Gender'] = train_set['Gender'].str.replace('Fe Male', 'Female')

# MonthlyIncome (Designation별 평균으로 변경)
train_set['MonthlyIncome'] = train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('median'))
test_set['MonthlyIncome'] = test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('median'))

# Occupation / freelancer -> 최빈값(Salaried)으로 변경
# print(train_set['Occupation'].value_counts())
# train_set['Occupation'] = train_set['Occupation'].str.replace('Free Lancer', train_set['Occupation'].mode())
train_set['Occupation'] = train_set['Occupation'].str.replace('Free Lancer', 'Salaried')
test_set['Occupation'] = test_set['Occupation'].str.replace('Free Lancer', 'Salaried')

# MarialStatus의 Unmarried, Divorced -> Single로 변경
# print(train_set['MaritalStatus'].value_counts())
# Married      949
# Divorced     375
# Single       349
# Unmarried    282
# Name: MaritalStatus, dtype: int64
# train_set['MaritalStatus'] = train_set['MaritalStatus'].str.replace('Divorced', 'Single')
# test_set['MaritalStatus'] = test_set['MaritalStatus'].str.replace('Divorced', 'Single')
# train_set['MaritalStatus'] = train_set['MaritalStatus'].str.replace('Unmarried', 'Single')
# test_set['MaritalStatus'] = test_set['MaritalStatus'].str.replace('Unmarried', 'Single')




train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)

train_set['Age']=np.round(train_set['Age'], 0).astype(int)
test_set['Age']=np.round(test_set['Age'], 0).astype(int)

train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)

train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)


# train_set['AgeBand'] = pd.cut(train_set['Age'], 5)
# 임의로 5개 그룹을 지정
# print(train_set['AgeBand'])
# [(17.957, 26.6] < (26.6, 35.2] < (35.2, 43.8] <
# (43.8, 52.4] < (52.4, 61.0]]
combine = [train_set, test_set]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 26.6, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 26.6) & (dataset['Age'] <= 35.2), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 35.2) & (dataset['Age'] <= 43.8), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 43.8) & (dataset['Age'] <= 52.4), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 52.4, 'Age'] = 4
# train_set = train_set.drop(['AgeBand'], axis=1)

'''
# combine = [train_set, test_set] # 10/20대, 30/40대, 50/60대로 나누기
# for dataset in combine:    
#     dataset.loc[ dataset['Age'] <= 29, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 2
#     dataset.loc[ dataset['Age'] > 49, 'Age'] = 3


# combine = [train_set, test_set] # 10/20/30, 40, 50/60대로 나누기
# for dataset in combine:    
#     dataset.loc[ dataset['Age'] <= 39, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 1
#     # dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 49), 'Age'] = 2
#     dataset.loc[ dataset['Age'] > 49, 'Age'] = 2

# combine = [train_set, test_set] # 10/20, 30, 40, 50, 60대로 나누기
# for dataset in combine:    
#     dataset.loc[ dataset['Age'] <= 29, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 2
#     dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 3
#     dataset.loc[ dataset['Age'] > 59, 'Age'] = 4

combine = [train_set, test_set] # 10/20, 30, 40, 50/60대로 나누기
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 29, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    # dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 49, 'Age'] = 4
'''


train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)

train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)


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



# 아웃라이어 확인
def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
                     
                           
# print(train_set['Designation'].unique())

# Age_out_index= outliers(train_set['Age'])[0]
TypeofContact_out_index= outliers(train_set['TypeofContact'])[0] # 0
CityTier_out_index= outliers(train_set['CityTier'])[0] # 0
DurationOfPitch_out_index= outliers(train_set['DurationOfPitch'])[0] #44
Gender_out_index= outliers(train_set['Gender'])[0] # 0
NumberOfPersonVisiting_out_index= outliers(train_set['NumberOfPersonVisiting'])[0] # 1
NumberOfFollowups_out_index= outliers(train_set['NumberOfFollowups'])[0] # 0
ProductPitched_index= outliers(train_set['ProductPitched'])[0] # 0
PreferredPropertyStar_out_index= outliers(train_set['PreferredPropertyStar'])[0]  # 0
MaritalStatus_out_index= outliers(train_set['MaritalStatus'])[0] # 0
NumberOfTrips_out_index= outliers(train_set['NumberOfTrips'])[0] # 38
Passport_out_index= outliers(train_set['Passport'])[0] # 0
PitchSatisfactionScore_out_index= outliers(train_set['PitchSatisfactionScore'])[0] # 0
OwnCar_out_index= outliers(train_set['OwnCar'])[0] # 0
NumberOfChildrenVisiting_out_index= outliers(train_set['NumberOfChildrenVisiting'])[0] # 0
Designation_out_index= outliers(train_set['Designation'])[0] # 89
MonthlyIncome_out_index= outliers(train_set['MonthlyIncome'])[0] # 138
# print(len(Designation_out_index))
lead_outlier_index = np.concatenate((#Age_out_index,                            # acc : 0.8650306748466258
                                    #  TypeofContact_out_index,                 # acc : 0.8920454545454546
                                    #  CityTier_out_index,                      # acc : 0.8920454545454546
                                     DurationOfPitch_out_index,               # acc : 0.9156976744186046
                                    #  Gender_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfPersonVisiting_out_index,        # acc : 0.8835227272727273
                                    #  NumberOfFollowups_out_index,             # acc : 0.8942598187311178
                                    #  ProductPitched_index,                    # acc : 0.8920454545454546
                                    #  PreferredPropertyStar_out_index,         # acc : 0.8920454545454546
                                    #  MaritalStatus_out_index,                 # acc : 0.8920454545454546
                                    #  NumberOfTrips_out_index,                 # acc : 0.8670520231213873
                                    #  Passport_out_index,                      # acc : 0.8920454545454546
                                    #  PitchSatisfactionScore_out_index,        # acc : 0.8920454545454546
                                    #  OwnCar_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfChildrenVisiting_out_index,      # acc : 0.8920454545454546
                                    #  Designation_out_index,                   # acc : 0.8869047619047619
                                    #  MonthlyIncome_out_index                  # acc : 0.8932926829268293
                                     ),axis=None)
# print(len(lead_outlier_index)) #577
# print(lead_outlier_index)

lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
# print(train_set_clean)


# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()

# train_set[['Age', 'DurationOfPitch']] = scaler.fit_transform(train_set[['Age', 'DurationOfPitch']])
# test_set[['Age', 'DurationOfPitch']] = scaler.transform(test_set[['Age', 'DurationOfPitch']])

x = train_set_clean.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups', 
                        #   'NumberOfTrips',
                          'Designation',
                          ], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups', 
                        #   'NumberOfTrips',
                          'Designation',
                          ], axis=1)

# x = train_set_clean.drop(['ProdTaken', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                             'OwnCar', 'MonthlyIncome',  'NumberOfFollowups'], axis=1)
# test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                             'OwnCar', 'MonthlyIncome',  'NumberOfFollowups'], axis=1)


y = train_set_clean['ProdTaken']
# print(x.shape)

'''
# ########################################################################################
# 아웃라이어 확인
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    # print("1사분위 :", quartile_1)
    # print("q2 :", q2)
    # print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    # print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | # or
                    (data_out<lower_bound))

# ### 슬라이싱
outliers_loc = outliers(train_set)
# print("이상치의 위치 :", outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(outliers_loc)
# plt.show()

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.25) # 25% 이상의 값을 이상치로 인식하도록 설정

outliers.fit(train_set)
results = outliers.predict(train_set)
# print(results) # [ 1  1  1 ... -1  1  1]


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

train_set[['Age', 'DurationOfPitch']] = scaler.fit_transform(train_set[['Age', 'DurationOfPitch']])
test_set[['Age', 'DurationOfPitch']] = scaler.transform(test_set[['Age', 'DurationOfPitch']])


# 모든 데이터 처리 완료 확인
# print(train_set.info())

# train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                             'OwnCar', 'MonthlyIncome',  'NumberOfFollowups'], axis=1)
# test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
#                           'OwnCar', 'MonthlyIncome',  'NumberOfFollowups'], axis=1)

train_set = train_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups', 
                        #   'NumberOfTrips',
                          'Designation',
                          ], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups', 
                        #   'NumberOfTrips',
                          'Designation',
                          ], axis=1)


x = train_set.drop(['ProdTaken'],axis=1) #axis는 컬럼 
# print(x) 

y = train_set['ProdTaken']
# print(y.shape)

# print(train_set.columns)

########################################################################################
'''

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE

x_train,x_test,y_train,y_test = train_test_split(
  x, y, train_size=0.91, shuffle=True, random_state=1234, stratify=y)

# x_train,x_test,y_train,y_test = train_test_split(
#   train_set, test_set, train_size=0.91, shuffle=True, random_state=1234, stratify=y)

n_splits = 6
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 범위 설정
# parameters = {'n_estimators': (100, 1000),
#               'depth': (3, 12),
#               'learning_rate': (0.01, 0.05),
#               'l2_leaf_reg': (1, 10),
#               'model_size_reg': (0, 10),
#               'od_pval' : (0, 10)
# }
# parameters = {'n_estimators': (100, 1000),
#               'depth': (4, 15),
#               'learning_rate': (0.01, 0.05),
#               'l2_leaf_reg': (1, 10),
#               'model_size_reg': (0, 10),
#               'od_pval' : (0, 10)
# }

# 최적의 매개변수
# parameters = {'n_estimators' : [1000],
#               'learning_rate' : [0.01],
#               'depth': [15],
#               'l2_leaf_reg' : [1],
#               'model_size_reg' : [0],
#               'od_pval' : [0],
# }

# 2. 모델
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score,accuracy_score
from catboost import CatBoostRegressor, CatBoostClassifier
from bayes_opt import BayesianOptimization

cat_params = {"learning_rate" : [0.20909079092170735],
                'depth' : [8],
                'od_pval' : [0.236844398775451],
                'model_size_reg': [0.30614059763442997],
                'l2_leaf_reg' :[5.535171839105427]}
# cat_params = {"learning_rate" : [0.6],
#                 'depth' : [8],
#                 'od_pval' : [0.9],
#                 'model_size_reg': [0.3],
#                 'l2_leaf_reg' :[5.446169770534698]}

cat = CatBoostClassifier(random_state=123, verbose=False, n_estimators=500)
model = RandomizedSearchCV(cat, cat_params, cv=kfold, n_jobs=-1)
# model = RandomizedSearchCV(cat, parameters, cv=kfold, n_jobs=-1)

# import time 
# start_time = time.time()
model.fit(x_train, y_train)   
# end_time = time.time()-start_time 

y_predict = model.predict(x_test)
results = accuracy_score(y_test, y_predict)
print('최적의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)
print('acc :', results)
# print('걸린 시간 :', end_time)

'''
y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)

submission = pd.read_csv(path + 'sample_submission.csv')
submission['ProdTaken'] = y_summit
submission.to_csv(path + 'sample_submission_70_01.csv', index=False)
'''

########################################################################################
# kfold 123 / cat random state 123 / tts 1234 / train_size=0.91
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9045300878972279
# acc : 0.9534883720930233
# 걸린 시간 : 4.618348836898804  --- 0.9241261722

# 70_01
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9051087777910354
# acc : 0.9534883720930233
# 70_01
# # cat = CatBoostClassifier(random_state=123
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9068349043471343
# acc : 0.9651162790697675  --- 0.8994032396


# 70_01
# cat = CatBoostClassifier(random_state=1234
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}        
# 최상의 점수 :  0.9085610309032335 
# acc : 0.9709302325581395 --- 0.9002557545

########################################################################################

# 70_2 parameter 2
# 최적의 매개변수 :  {'od_pval': 0.9, 'model_size_reg': 0.3, 'learning_rate': 0.6, 'l2_leaf_reg': 5.446169770534698, 'depth': 8}
# 최상의 점수 :  0.9051087777910353
# acc : 0.9651162790697675

# 70_03 이상치 제거(기본 25%)
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}        
# 최상의 점수 :  0.9123119331452666
# acc : 0.9545454545454546 - 0.8976982097

########################################################################################

# kfold 666 / cat random state 123 / tts 1234 / train_size=0.91
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9114445372469474
# acc : 0.9534883720930233
# 걸린 시간 : 4.776421308517456

# kfold 666 / cat random state 666 / tts 1234 / train_size=0.91
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9091476752972993
# acc : 0.9534883720930233
# 걸린 시간 : 5.033710956573486

# kfold 666 / cat random state 666 / tts 12345 / train_size=0.91
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9137493536968541
# acc : 0.9418604651162791
# 걸린 시간 : 4.838541030883789

# kfold 666 / cat random state 666 / tts 1234 / train_size=0.90
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9051700332188136
# acc : 0.953125
# 걸린 시간 : 4.3536536693573

# kfold 666 / cat random state 666 / tts 1234 / train_size=0.92
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9118316268486918
# acc : 0.954248366013072
# 걸린 시간 : 4.6465513706207275

###################################################################
# kfold 666 / cat random state 666 / tts 1234 / train_size=0.93
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.907716428549762
# acc : 0.9552238805970149
# 걸린 시간 : 4.655333995819092
###################################################################

# kfold 666 / cat random state 666 / tts 1234 / train_size=0.94
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9103437383872167
# acc : 0.9478260869565217
# 걸린 시간 : 4.7329792976379395

###################################################################
# kfold 666 / cat random state 666 / tts 1234 / train_size=0.93 / 컬럼삭제 N.O.T 추가
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9048935298935299
# acc : 0.9626865671641791
# 걸린 시간 : 4.645659446716309
###################################################################

# submission 50_01
# kfold 666 / cat random state 123 / tts 1234 / train_size=0.93 / 컬럼삭제 N.O.T 추가
# 직업 freelancer -> 최빈값(Salaried)으로 변경
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9060196560196561
# acc : 0.9701492537313433 --- 0.8925831202
# 걸린 시간 : 4.648986101150513

# submission 50_02
# 최적의 매개변수 :  {'od_pval': 0, 'n_estimators': 1000, 'model_size_reg': 0, 'learning_rate': 0.01, 'l2_leaf_reg': 1, 'depth': 15}
# 최상의 점수 :  0.9048935298935299
# acc : 0.9776119402985075 --- 0.9011082694

# 최적의 매개변수 :  {'od_pval': 10, 'n_estimators': 1000, 'model_size_reg': 10, 'learning_rate': 0.01, 'l2_leaf_reg': 1, 'depth': 12}
# 최상의 점수 :  0.903769299602633
# acc : 0.9626865671641791

# 최적의 매개변수 :  {'od_pval': 10, 'n_estimators': 1000, 'model_size_reg': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 1, 'depth': 15}
# 최상의 점수 :  0.9032043407043407
# acc : 0.9701492537313433

# 최적의 매개변수 :  {'od_pval': 10, 'n_estimators': 1000, 'model_size_reg': 0, 'learning_rate': 0.05, 'l2_leaf_reg': 1, 'depth': 10}
# 최상의 점수 :  0.9082700124366792
# acc : 0.9552238805970149


###################################################################

# submission 60_01
# 최적의 매개변수 :  {'od_pval': 0, 'n_estimators': 1000, 'model_size_reg': 0, 'learning_rate': 0.01, 'l2_leaf_reg': 1, 'depth': 15}
# 최상의 점수 :  0.9103410341034103
# acc : 0.9562043795620438
# 걸린 시간 : 158.08576011657715

# submission 60_02
# 최적의 매개변수 :  {'od_pval': 10, 'n_estimators': 1000, 'model_size_reg': 10, 'learning_rate': 0.01, 'l2_leaf_reg': 1, 'depth': 15}
# 최상의 점수 :  0.9103410341034103
# acc : 0.9562043795620438
# 걸린 시간 : 291.2138338088989


###################################################################
# cat params 사용
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}

# submission 60_03
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9092409240924092
# acc : 0.9635036496350365
# 걸린 시간 : 7.0717453956604

# submission 60_04 / 컬럼삭제했던 N.O.T 복구 
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9108910891089108
# acc : 0.9708029197080292 --- 0.9002557545

# submission 60_05 / 컬럼삭제했던 N.O.T 복구 (아웃라이어...)
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9161623744957078 --- 0.9011082694
# acc : 0.9552238805970149

# submission_60_06 / 컬럼삭제 N.O.T
# 최적의 매개변수 :  {'od_pval': 0.236844398775451, 'model_size_reg': 0.30614059763442997, 'learning_rate': 0.20909079092170735, 'l2_leaf_reg': 5.535171839105427, 'depth': 8}
# 최상의 점수 :  0.9060196560196561 --- 0.8925831202
# acc : 0.9701492537313433

###################################################################
# cat params 2 사용
# {'target': 0.9776119402985075, 'params': {'depth': 7.782828166355092, 'fold_permutation_block': 10.0, 'l2_leaf_reg': 5.446169770534698, 'learning_rate': 0.6, 'model_size_reg': 0.3, 'od_pval': 0.9}}

# submission 70_01
# 최적의 매개변수 : {'od_pval': 0.9, 'model_size_reg': 0.3, 'learning_rate': 0.6, 'l2_leaf_reg': 5.446169770534698, 'depth': 8}
# 최상의 점수 : 0.9009482967816301
# acc : 0.9477611940298507