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
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)

# parameters = [
#     {'RF__n_estimators': [100,200], 'RF__max_depth': [6,8], 'RF__min_samples_leaf': [3,5,7,10]},          # 32번
#     {'RF__max_depth': [6,8], 'RF__min_samples_leaf': [3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},    # 64번
#     {'RF__min_samples_leaf': [3,5,7,10], 'RF__n_jobs': [-1]}                                            # 12번 = 총 108번
# ]

parameters = [
    {'RF__n_estimators': [100,200,300], 'RF__max_depth': [4,6,8], 'RF__min_samples_leaf': [3,5,7,10]},          # 32번
    {'RF__max_depth': [6,8], 'RF__min_samples_leaf': [3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},    # 64번
    {'RF__min_samples_leaf': [3,5,7,10], 'RF__n_jobs': [-1]}                                            # 12번 = 총 108번
]


#2. 모델
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
# 두가지가 표기법만 다르고 성능은 동일

pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestRegressor())]
                , verbose=1) # 스케일러와 모델명


#3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingRandomSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=5, verbose=1)

model.fit(x_train, y_train) 
# pipeline에서 model.fit을 하면 scaling의 fit transform과 훈련의 fit이 같이 실행됨

#4. 평가, 예측
result = model.score(x_test, y_test) # pipeline의 score
print('model.score 결과 :', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2) 

# submission_04
# model.score 결과 : 0.06356930766762195
# r2 스코어 :  0.06356930766762195

# submission_04_2 / 파라미터 내용 추가 + train_size=0.75
# model.score 결과 : 0.05017487566301907
# r2 스코어 :  0.05017487566301907

# submission_04_3 / 스플릿 랜덤스테이트 666
# model.score 결과 : 0.05485985953624115
# r2 스코어 :  0.05485985953624115

# submission_04_4 / train_size=0.9
# model.score 결과 : 0.06350740680203135
# r2 스코어 :  0.06350740680203117

# submission_04_5 / 04에서 randomsearch 를 halvingrandomsearch로 변경
# model.score 결과 : 0.05101889106222252
# r2 스코어 :  0.05101889106222252

# submission_04_6 / 04에서 모든 랜덤스테이트 1234
# model.score 결과 : 0.06800684875657767    
# r2 스코어 :  0.06800684875657775

y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (39608, 14)


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'sample_submission_04_6.csv', index=False)