# 데이콘 문제풀기
# https://dacon.io/competitions/official/235927

from random import Random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터

path = 'D:/study_data/_data/dacon_radar/'
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

# print(train_set)
# print(train_set.shape) # (39607, 70)

# print(train_set.columns)

# print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
# print(train_set.describe())

# print(test_set)
# print(test_set.shape) # (39608, 56)

# # 결측치 확인 --- 없음
# print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
# print(test_set.isnull().sum())

train_x = train_set.filter(regex='X') # Input : X Featrue : 56
train_y = train_set.filter(regex='Y') # Output : Y Feature : 14


############################################################################################

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

# 1사분위 : 1.56
# q2 : 12.94
# 3사분위 : 69.884
# iqr : 68.324
# 이상치의 위치 : (array([    0,     0,     0, ..., 39606, 39606, 39606], dtype=int64), array([ 8, 45, 48, ...,  8, 45, 48], dtype=int64))
# 1사분위 : -25.912
# q2 : 0.962
# 3사분위 : 13.8
# iqr : 39.712
# 이상치의 위치 : (array([24992], dtype=int64), array([3], dtype=int64))

### 슬라이싱
outliers_loc = outliers(train_x)
print("이상치의 위치 :", outliers_loc)
outliers_loc = outliers(train_y)
print("이상치의 위치 :", outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(outliers_loc)
# plt.show()


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3) # 25% 이상의 값을 이상치로 인식하도록 설정

outliers.fit(train_x)
results = outliers.predict(train_x)
print(results)
outliers.fit(train_y)
results = outliers.predict(train_y)
print(results) 

############################################################################################


x_train, x_test, y_train, y_test = train_test_split(
    train_x, train_y, shuffle=True, random_state=123, train_size=0.85)
# print(train_x.shape,train_y.shape)  #(39607, 56) (39607, 14)     
# print(test_set.shape) # (39608, 56)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)



#2. 모델구성
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# model = MultiOutputRegressor(LinearRegression()).fit(x_train, y_train)
model = MultiOutputRegressor(RandomForestRegressor())

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('결과 :', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2) 

# 01_2 / random_state=1234
# 결과 : 0.03831989800064245
# r2 스코어 :  0.03831989800064245

# 01_3 / random_state=666
# 결과 : 0.040186523484182494
# r2 스코어 :  0.040186523484182494

# 01_4 / random_state=666, train_size=0.82
# 결과 : 0.04160440237706323
# r2 스코어 :  0.04160440237706323

# 01_5 / random_state=666, train_size=0.85 --- 1.9745652727
# 결과 : 0.04195296378694332
# r2 스코어 :  0.04195296378694332

# 01_6 / random_state=666, train_size=0.86
# 결과 : 0.0406536110812733
# r2 스코어 :  0.0406536110812733


############### 아웃라이어 적용 ###############

# 01_7 / random_state=666, train_size=0.86 , outlier .25 제거 --- 1.9745652727
# 결과 : 0.04195296378694332
# r2 스코어 :  0.04195296378694332

# 01_8 / random_state=666, train_size=0.86 , outlier .3 제거
# 결과 : 0.04195296378694332
# r2 스코어 :  0.04195296378694332


############### 기존 baseline2 파일에서 RandomForestRegressor 적용 ###############

# 01_20 / random_state=123, train_size=0.85 , outlier .3 제거




y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (39608, 14)


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'sample_submission_01_20.csv', index=False)
