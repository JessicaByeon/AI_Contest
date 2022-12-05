import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

#1. 데이터
path = 'D:/study_data/_data/dacon_travel/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

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

# print(train_set.info())
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
                          'NumberOfTrips',
                          'NumberOfFollowups',
                          ], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups', 
                          'NumberOfTrips',
                          'NumberOfFollowups',
                          ], axis=1)


x = train_set.drop(['ProdTaken'],axis=1) #axis는 컬럼 
# print(x) 

y = train_set['ProdTaken']
# print(y.shape)

# print(train_set.columns)

########################################################################################


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold


x_train, x_test, y_train, y_test = train_test_split(
  x, y, train_size=0.93, shuffle=True, random_state=1234, stratify=y)

# x_train,x_test,y_train,y_test = train_test_split(
#   train_set, test_set, train_size=0.91, shuffle=True, random_state=1234, stratify=y)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=666)

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
parameters = {'n_estimators' : [1000],
              'learning_rate' : [0.01],
              'depth': [15],
              'l2_leaf_reg' : [1],
              'model_size_reg' : [0],
              'od_pval' : [0],
}

# 2. 모델
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier


cat = CatBoostClassifier(random_state=123, verbose=False, n_estimators=500)
model = RandomizedSearchCV(cat, parameters, cv=kfold, n_jobs=-1)

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
submission.to_csv(path + 'sample_submission_50_02.csv', index=False)

'''

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