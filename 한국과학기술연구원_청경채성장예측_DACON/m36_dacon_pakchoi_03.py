# 데이콘 문제풀기
# https://dacon.io/competitions/official/235961/overview/description

import numpy as np
import pandas as pd
import glob
import os
import joblib as jb


#1. 데이터
path = 'D:/study_data/_data/dacon_pakchoi/'

train_data, label_data, val_data, val_target, test_input, test_target = jb.load(path+'datasets.dat')

# print(train_data[0])
# print(len(train_data), len(label_data)) # 1607 1607
# print(len(train_data[0]))   # 1440
# print(label_data)   # 1440
# print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
# print(val_data.shape) # (206, 1440, 37)
# print(val_target.shape) # (206,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    train_data, label_data, train_size=0.91, shuffle=False)


#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.layers import LSTM, GRU, Bidirectional, BatchNormalization

# model = Sequential()
# model.add(LSTM(50,input_shape=(1440, 37)))
# # model.add(GRU(50, activation='relu'))
# # model.add(GRU(50))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))
# # model.summary()

model = Sequential()
model.add(Bidirectional(LSTM(100, input_shape=(1440, 37))))
# model.add(GRU(100, input_shape=(1440, 37)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# model.summary() 


#3. 컴파일, 훈련
import time
start_time = time.time()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', verbose=1, factor=0.5)

earlyStopping =EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=500, batch_size=1000, 
                 validation_split=0.2, callbacks=[earlyStopping, reduce_lr], verbose=1)


model.save_weights('D:\study_data\_save\_h5/dacon_packchoi_saveweights_02.h5')

model.save('D:\study_data\_save\_h5/dacon_packhoi_savemodel_02.h5')
# model = load_model('D:\study_data\_save\_h5/dacon_packhoi_01.h5')

end_time = time.time()-start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)
rmse = np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE :', rmse)

#5. 제출
# test_pred -> TEST_files
test_pred = model.predict(test_input)
for i in range(6):
    thislen=0
    thisfile = 'D:\study_data\_data\dacon_pakchoi/test_target/'+'TEST_0'+str(i+1)+'.csv'
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = test_pred[thislen:thislen+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    thislen+=len(test['rate'])
    
# TEST_files -> zip file
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_pakchoi/test_target")
with zipfile.ZipFile("submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
    
print('완료')
print('걸린시간:', end_time)
