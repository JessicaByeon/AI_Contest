# 데이콘 문제풀기
# https://dacon.io/competitions/official/235927

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

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

# 결측치 확인 --- 없음
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
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

from sklearn.model_selection import KFold
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)



#2. 모델구성
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor

# model = MultiOutputRegressor(LinearRegression()).fit(x_train, y_train)
model = MultiOutputRegressor(Ridge(random_state=123)).fit(x_train, y_train)


#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('결과 :', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2) 


#_02_2
# MultiOutputRegressor(Ridge(random_state=123))
# 결과 : 0.03857580574354289
# r2 스코어 :  0.03857580574354289

# _02
# MultiOutputRegressor(LinearRegression()) + minmax scaler (+kfold 결과값 동일)
# 결과 : 0.03845767854102239
# r2 스코어 :  0.03845767854102239

# _01_2
# MultiOutputRegressor(LinearRegression()) (베이스라인)
# 결과 : 0.03831989800064245
# r2 스코어 :  0.03831989800064245



y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (39608, 14)


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'sample_submission_02_2.csv', index=False)
