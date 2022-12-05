# 데이콘 문제풀기
# https://dacon.io/competitions/official/235961/overview/description

import numpy as np
import pandas as pd
import glob
import os



#1. 데이터
path = 'D:/study_data/_data/dacon_pakchoi/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

train_data, label_data = aaa(train_input_list, train_target_list) #, False)

print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # [0.5     0.66667 0.6     ... 0.01061 0.03615 0.03358]
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)




#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Flatten, Conv1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(1440, 37)))
# model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# model.summary() 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=500, batch_size=100, 
                 validation_split=0.2, callbacks=[earlyStopping], verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


#5. 제출
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_pakchoi/test_target")
with zipfile.ZipFile("submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
    
    
# # autokeras tunning 시..  사용... 
# https://wandb.ai/authors/bcnn/reports/W-B-Keras-Tuner--Vmlldzo0NTU4ODM   

# tuner = RandomSearch(
#   hypermodel =  build_model,
#   max_trials = 5,
#   executions_per_trial = 6,
#   hyperparameters = hp,
#   objective = 'mse',
#   ...
# )

# tuner %>% fit_tuner(x = x, y = y, 
#                     epochs = 100, 
#                     validation_data = list(x_val, y_val))