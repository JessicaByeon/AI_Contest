import pandas as pd
import numpy as np
import glob
# from keras.models import Sequential
# from keras.layers import Dense,LSTM,GRU
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,LSTM,GRU
# from keras.layers import Bidirectional,Dense,LSTM
from sklearn.model_selection import train_test_split
train_data = np.load('D:/study_data/_save/_npy/train_data12.npy')
label_data = np.load('D:/study_data/_save/_npy/label_data12.npy')
val_data = np.load('D:/study_data/_save/_npy/val_data12.npy')
val_target = np.load('D:/study_data/_save/_npy/val_target12.npy')
test_data = np.load('D:/study_data/_save/_npy/test_data12.npy')
test_target = np.load('D:/study_data/_save/_npy/test_target12.npy')
print(train_data.shape,label_data.shape)    # (1607, 1440, 37) (1607,)
print(val_data.shape,val_target.shape)      # (206, 1440, 37) (206,)
print(test_data.shape,test_target.shape)    # (195, 1440, 37) (195,)
# train_data = train_data.reshape(1607, 1440, 37, 1)
# val_data = val_data.reshape(206, 1440, 37, 1)
# test_data = test_data.reshape(195, 1440, 37, 1)
                                                  
# x_train,x_test,y_train,y_test = train_test_split(train_data,label_data,train_size=0.87,shuffle=False)
# print(x_train.shape,x_test.shape) # (1462, 1440, 37) (145, 1440, 37)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = StandardScaler()
# x_train = x_train.reshape(1462,53280)
# x_test = x_test.reshape(145,53280)

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# x_train = x_train.reshape(1462,1440,37)
# x_test = x_test.reshape(145,1440,37)

#2. 모델구성      
                                                                                              
model = Sequential()
model.add(Bidirectional(GRU(50,return_sequences=True),input_shape=(1440,37)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()
   
import time
start_time = time.time()
#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import time
es = EarlyStopping(monitor='val_loss',patience=5,mode='min',verbose=1)


model.compile(loss='mae', optimizer='adam',metrics=['mse'])
# "".join은 " "사이에 있는 문자열을 합치겠다는 기능
hist = model.fit(train_data, label_data, epochs=20, batch_size=2000, 
                validation_data=(val_data, val_target),
                verbose=2,callbacks = [es]
                )
model.save_weights("C:\Study\_save/keras57_10_save_weights1.h5")


#4. 평가,예측
# loss = model.evaluate(x_test, y_test)
# print('loss :', loss)
# from sklearn.metrics import r2_score
# y_predict = model.predict(x_test)
# r2 = r2_score(y_predict,y_test)
# from sklearn.metrics import mean_squared_error
# rmse = np.sqrt(mean_squared_error(y_test,y_predict))
                      
                  
# y_predict = model.predict(x_test)
# print(y_test.shape) #(152,)
# print(y_predict.shape) #(152, 13, 1)

# from sklearn.metrics import accuracy_score, r2_score,accuracy_score
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 :', r2)


# model.fit(train_data,label_data)
y_summit = model.predict(test_data)
model.save_weights("C:\Study\_save/keras57_15_save_weights1.h5")
path2 = 'D:\study_data\_data\_csv\dacon_grow\\test_target/' # ".은 현재 폴더"
targetlist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv','TEST_06.csv']
# [29, 35, 26, 32, 37, 36]
empty_list = []
for i in targetlist:
    test_target2 = pd.read_csv(path2+i)
    empty_list.append(test_target2)
    
empty_list[0]['rate'] = y_summit[:29]
empty_list[0].to_csv(path2+'TEST_01.csv')
empty_list[1]['rate'] = y_summit[29:29+35]
empty_list[1].to_csv(path2+'TEST_02.csv')
empty_list[2]['rate'] = y_summit[29+35:29+35+26]
empty_list[2].to_csv(path2+'TEST_03.csv')
empty_list[3]['rate'] = y_summit[29+35+26:29+35+26+32]
empty_list[3].to_csv(path2+'TEST_04.csv')
empty_list[4]['rate'] = y_summit[29+35+26+32:29+35+26+32+37]
empty_list[4].to_csv(path2+'TEST_05.csv')
empty_list[5]['rate'] = y_summit[29+35+26+32+37:]
empty_list[5].to_csv(path2+'TEST_06.csv')
# submission = submission.fillna(submission.mean())
# submission = submission.astype(int)

import os
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\_csv\dacon_grow/test_target")
with zipfile.ZipFile("D:\study_data\_data\_csv\dacon_grow/sample_submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
print('Done')
print('R2 :', r2)
print('RMSE :', rmse)
end_time = time.time()-start_time
print('걸린 시간:', end_time)
# Minmax
# loss : [0.4065817594528198, 0.006896551698446274]
# R2 : -46.606687007464885
# RMSE : 1.6205276408795766
# 걸린 시간: 1512.3487508296967