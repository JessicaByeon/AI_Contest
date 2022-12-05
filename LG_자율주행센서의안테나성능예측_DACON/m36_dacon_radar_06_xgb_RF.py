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

allfeature = round(train_x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))

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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]


#3. 훈련, 평가, 예측
for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGBRegressor 의 스코어 : ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
    
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
    
    x_modified = np.delete(train_x, featurelist, axis=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_modified, y, shuffle=True, train_size=0.8, random_state=123)
    model.fit(x_train2, y_train2) # 삭제 한 후의 데이터로 다시 데이터를 나눠줌
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGBRegressor 의 데이터 삭제 후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 데이터 삭제 후 스코어: ', score)

# 자를 갯수:  11
# DecisionTreeRegressor 의 스코어:  -0.908691015665056





'''
#4. 평가, 예측
result = model.score(x_test2, y_test2)
print("model.score : ", result)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test2)
r2 = r2_score(y_test2, y_predict)
print('r2_score : ', r2)



y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (39608, 14)


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'sample_submission_06.csv', index=False)
'''