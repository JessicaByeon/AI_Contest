import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
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

from sklearn.model_selection import KFold
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,8,10,12], 'min_samples_leaf': [3,5,7,10]},          # 32번
    {'max_depth': [6,8,10,12], 'min_samples_leaf': [3,5,7,10], 'min_samples_split': [2,3,5,10]},    # 64번
    {'min_samples_leaf': [3,5,7,10], 'min_samples_split': [2,3,5,10], 'n_jobs': [-1,2,4]}           # 48번 = 총 144번
]


#2. 모델구성
from sklearn.ensemble import RandomForestRegressor

# model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,     # 42*5=210번 실행 : Fitting 5 folds for each of 42 candidates, totalling 210 fits
#                      refit=True, n_jobs=-1) # refit=True : 최적의 모델을 찾아내기 위해서 사용, n_jobs cpu 사용갯수(-1 : 전체를 다 쓰겠다는 뜻)
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
                     refit=True, n_jobs=-1)


#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_) # 가장 좋은 추정치
print("최적의 파라미터  : ", model.best_params_)
print('best_score : ', model.best_score_) # 정확도
print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  :', r2_score(y_test, y_pred_best))

print('걸린시간 :', np.round(end-start, 4), "초")


y_summit = model.best_estimator_.predict(test_set)
print(y_summit)
print(y_summit.shape) # (39608, 14)


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'sample_submission_03.csv', index=False)

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=5, min_samples_split=3, n_jobs=4)
# 최적의 파라미터  :  {'n_jobs': 4, 'min_samples_split': 3, 'min_samples_leaf': 5}
# best_score :  0.07163721833024138
# model.score :  0.07101063764744336
# r2_score :  0.07101063764744339
# 최적 튠  : 0.07101063764744349
# 걸린시간 : 559.8403 초