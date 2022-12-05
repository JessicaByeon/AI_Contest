# feature importances가 전체 중요도에서 하위 20% 컬럼들을 제거

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
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

# 결측치 확인 / 각 컬럼당 null 갯수 확인 --- 없음
print(train_set.isnull().sum())
print(test_set.isnull().sum())


train_x = train_set.filter(regex='X') # Input : X Featrue : 56
train_y = train_set.filter(regex='Y') # Output : Y Feature : 14

##########################################################
train_x= np.delete(train_x, [3, 22, 40, 47], axis=1)
##########################################################

print(train_x.shape, train_y.shape)  #(39607, 56) (39607, 14)

'''
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
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", result)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score : ', r2)

# model.score :  0.02150625430997025
# r2_score :  0.02150625430997025

print("=====================================")
print(model, ':', model.feature_importances_)
# [0.00774599 0.00717961 0.01866974 0.         0.02553585 0.01341922
#  0.03963191 0.01425418 0.02970484 0.01556171 0.00711519 0.01535477   
#  0.01971818 0.01721548 0.02096218 0.01825893 0.02032897 0.02025386   
#  0.02344234 0.02275957 0.02573416 0.01987811 0.         0.01421158   
#  0.01905589 0.01554514 0.0188824  0.0216203  0.0210056  0.01589746   
#  0.01483581 0.03872157 0.01734604 0.01835705 0.01556167 0.01424055   
#  0.01851    0.01772552 0.01647735 0.01812856 0.01663916 0.02065101   
#  0.01405167 0.01867788 0.01888126 0.02079523 0.         0.
#  0.03546653 0.01747614 0.01707677 0.01958    0.01850887 0.02031136   
#  0.01859803 0.02443882]
'''

'''
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (39608, 14)


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'sample_submission_05.csv', index=False)
'''