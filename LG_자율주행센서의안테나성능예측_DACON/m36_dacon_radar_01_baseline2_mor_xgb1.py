# 데이콘 문제풀기
# https://dacon.io/competitions/official/235927

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

print(train_set)
print(train_set.shape) # (39607, 70)

print(train_set.columns)
# Index(['X_01', 'X_02', 'X_03', 'X_04', 'X_05', 'X_06', 'X_07', 'X_08', 'X_09',
#        'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18',
#        'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27',
#        'X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36',
#        'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44', 'X_45',
#        'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52', 'X_53', 'X_54',
#        'X_55', 'X_56', 'Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07',
#        'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14'],
#       dtype='object')

print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
# non-null / dtypes: float64(65), int64(5)


print(train_set.describe())
#                X_01          X_02          X_03  ...          Y_12          Y_13
# Y_14
# count  39607.000000  39607.000000  39607.000000  ...  39607.000000  39607.000000  39607.000000
# mean      68.412040    103.320166     68.826354  ...    -26.237762    -26.233869    -26.245868
# std        2.655983      0.000372      5.151167  ...      0.656329      0.655090      0.655989
# min       56.268000    103.320000     56.470000  ...    -29.544000    -29.448000    -29.620000
# 25%       66.465000    103.320000     65.070000  ...    -26.630000    -26.624000    -26.640000
# 50%       68.504000    103.320000     67.270000  ...    -26.198000    -26.193000    -26.204000
# 75%       69.524000    103.320000     71.770000  ...    -25.799000    -25.794000    -25.809000
# max       84.820000    103.321000     89.170000  ...    -23.722000    -23.899000    -23.856000
# [8 rows x 70 columns]

print(test_set)
print(test_set.shape) # (39608, 56)

# 결측치 확인 --- 없음
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
print(test_set.isnull().sum())


# x = train_set.drop(['Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07', 
#                     'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14'], axis=1) #axis는 컬럼 
# print(x) #(39607, 56)
#        Age  TypeofContact  ...  Designation  MonthlyIncome
# id                         ...
# 1     28.0              0  ...    Executive        20384.0
# 2     34.0              1  ...      Manager        19599.0
# 3     45.0              0  ...      Manager        22295.0
# 4     29.0              0  ...    Executive        21274.0
# 5     42.0              1  ...      Manager        19907.0
# ...    ...            ...  ...          ...            ...
# 1951  28.0              1  ...    Executive        20723.0
# 1952  41.0              1  ...          AVP        31595.0
# 1953  38.0              0  ...    Executive        21651.0
# 1954  28.0              1  ...      Manager        22218.0
# 1955  22.0              0  ...    Executive        17853.0
# [1955 rows x 18 columns]

# y = np.array(['Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07', 
#                'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14'])
# y = train_set[y]
# print(y.shape) # (39607, 14)

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
outliers = EllipticEnvelope(contamination=.25) # 25% 이상의 값을 이상치로 인식하도록 설정

outliers.fit(train_x)
results = outliers.predict(train_x)
print(results)
outliers.fit(train_y)
results = outliers.predict(train_y)
print(results) 

############################################################################################


x_train, x_test, y_train, y_test = train_test_split(
    train_x, train_y, shuffle=True, random_state=666, train_size=0.85)
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
from xgboost import XGBRegressor

model = MultiOutputRegressor(XGBRegressor())


#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('결과 :', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2) 


############# baseline2 에 xgb 적용 + 이상치 제거 #############
# 01_11 / .25 제거
# 결과 : 0.04266305773271413
# r2 스코어 :  0.04266305773271413


############# baseline2 에 xgb 적용 #############

# 01_10 / random state = 666 / train_size = 0.85 --- 2.0004647211
# 결과 : 0.04266305773271413
# r2 스코어 :  0.04266305773271413


############# baseline2 #############

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


y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (39608, 14)

# submission = pd.read_csv(path + 'sample_submission.csv')
# # submission['Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07', 
# #            'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14'] = y_summit
# # submission.to_csv(path + 'sample_submission_01.csv', index=True)

# for idx, col in enumerate(submission.columns):
#     if col=='ID':
#         continue
#     submission[col] = y_summit[:,idx-1]
# print('Done.')
# submission.to_csv('sample_submission_01.csv',index=False)


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'sample_submission_01_11.csv', index=False)
