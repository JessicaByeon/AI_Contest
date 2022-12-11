import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import catboost as cgb
import optuna
from optuna import Trial,visualization
from optuna.samplers import TPESampler,RandomSampler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
path = 'D:\\ai_data\AI_FACTORY\kogas/'

data_all = pd.read_csv(path+'Total_data.csv')
mean_data = np.load(path+'mean_data.npy')
# print(data_all.head())

# heat_map = data_all
# plt.figure(figsize=(12,8))
# plt.title("Feature 상관관계 시각화",y = 1.05, size = 15)
# sns.heatmap(heat_map.astype(float).corr(),linewidths=0.1,vmax=1.0,
#             square = True, cmap='PuBuGn',linecolor="white",annot=True,annot_kws={"size":7})

# print(data_all.columns)
# Index(['YEAR', 'MONTH', 'CIVIL', 'IND', 'Total', 'CIVILper', 'INDper', 'RP',
#        'GAS_PRICE', 'OIL_PRICE', 'MFG', 'Meantemp', 'nanbangdoil',
#        'naengbangdoil', 'lowtemp', 'hightemp', 'amount_of_gas', 'INDcon',
#        'CIVILcon', 'QUARTER_Q1', 'QUARTER_Q2', 'QUARTER_Q3', 'QUARTER_Q4',
#        'MOM_CIVIL', 'MOM_IND'],
#       dtype='object')

data_ck = data_all[['RP']]
idx = pd.date_range('01-01-1996','12-31-2020',freq='M')
data_ck['Date'] = pd.to_datetime(idx)
data_ck = data_ck.set_index('Date',drop=True)
data_ck.plot() # 인덱스를 Datetime 포맷으로 설정했다면 이 명령어로 실행하면 보기 쉽다.
# 데이터 양이 방대하다면 일간,주간,월간으로 자체 슬라이싱해서 보는 것도 좋을 듯하다
plt.show()


# print(data_ck.head())

# plt.show()
#### 데이터 분포 확인 수치형
# sns.lmplot(data = data_all,x = 'CIVIL',y='IND')
# sns.pairplot(data_all)

# plt.show()
# size = 24
# def split_x(dataset,size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         subset = dataset[i : (i + size)]
#         aaa.append(subset)
#     return np.array(aaa)
# import matplotlib.pyplot as plt
# import seaborn as sns
# y = data_all[['도시가스(톤)_민수용','도시가스(톤)_산업용']]
# x = data_all
# ###### 분포도 확인
# plt.figure(figsize=(8,6))
# plt.title('y라벨 분포도')
# sns.scatterplot(x = y.index, y = y['도시가스(톤)_민수용'])
# plt.show()


# ###### 변수 변화
# plt.figure(figsize=(8,6))
# plt.title('민수용 변화')
# sns.lineplot(x=pd.to_datetime(y['MONTH']),y = y['도시가스(톤)_민수용'])
# plt.show()


'''
x1 = ccc[:,:12]
y1 = bbb[:,12:]
print(x1,x1.shape)
x1 = x1.reshape(x1.shape[0]*x1.shape[1],x1.shape[2])
y1 = y1.reshape(y1.shape[0]*y1.shape[1],y1.shape[2])
# print(x1.shape)
# print(y1.shape)

from sklearn.model_selection import RandomizedSearchCV, train_test_split,KFold
from keras.layers import LayerNormalization,Conv1D
x_train,x_test,y_train,y_test = train_test_split(
    x1,y1,train_size=0.9,shuffle=False)
n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=123)
'''

'''
optuna.trial.Trial.suggest_categorical() : 리스트 범위 내에서 값을 선택한다.
optuna.trial.Trial.suggest_int() : 범위 내에서 정수형 값을 선택한다.
optuna.trial.Trial.suggest_float() : 범위 내에서 소수형 값을 선택한다.
optuna.trial.Trial.suggest_uniform() : 범위 내에서 균일분포 값을 선택한다.
optuna.trial.Trial.suggest_discrete_uniform() : 범위 내에서 이산 균일분포 값을 선택한다.
optuna.trial.Trial.suggest_loguniform() : 범위 내에서 로그 함수 값을 선택한다.
'''
from sklearn.metrics import r2_score,mean_absolute_error
''''
def objectiveCAT(trial: Trial, x_train, y_train, x_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 200, 1000),
        'depth' : trial.suggest_int('depth', 8, 16),
        'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256),
        'learning_rate' : trial.suggest_float('learning_rate', 0, 1),
        'od_pval' : trial.suggest_float('od_pval', 0, 1),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 4),
        'random_state' :trial.suggest_int('random_state', 1, 2000)
    }
    
    # 학습 모델 생성
    model = CatBoostRegressor(**param,loss_function='MultiRMSE')
    # model = CatBoostRegressor(**param,task_type='GPU')
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    
    # 모델 성능 확인
    score = r2_score(CAT_model.predict(x_test), y_test)
    
    return score
# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

# n_trials 지정해주지 않으면, 무한 반복
study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 2)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))

# Trial 1 finished with value: 0.9807730244142494 and parameters: 
# {'n_estimators': 777, 'depth': 9, 'fold_permutation_block': 123,
# 'learning_rate': 0.41656368743218575, 'od_pval': 0.6109150578820269, 
# 'l2_leaf_reg': 3.81312375568168, 'random_state': 83}. 

# Best is trial 1 with value: 0.9807730244142494.
# Best trial : score 0.9807730244142494,
# params {'n_estimators': 777, 'depth': 9, 
# 'fold_permutation_block': 123, 'learning_rate': 0.41656368743218575, 
# 'od_pval': 0.6109150578820269, 'l2_leaf_reg': 
# 3.81312375568168, 'random_state': 83}

# # y123 = y_pred[69,:]

parameters = {'n_estimators': [777], 
              'depth': [9], 
              'fold_permutation_block': [123], 
              'learning_rate': [0.41656368743218575], 
              'od_pval': [0.6109150578820269], 
              'l2_leaf_reg': [3.81312375568168], 
              'random_state': [83]}
# r2 = r2_score(y_test,y_pred)
# print(r2)

cat = cgb.CatBoostRegressor(loss_function='MultiRMSE')
model = RandomizedSearchCV(cat,parameters,cv=kfold,n_jobs=-1)


model.fit(x_train,y_train)

y_pred = model.predict(x_test)

r2 =r2_score(y_test,y_pred)
print('r2 :',r2)
mae = mean_absolute_error(y_test,y_pred)
print('MAE :',mae)
'''




