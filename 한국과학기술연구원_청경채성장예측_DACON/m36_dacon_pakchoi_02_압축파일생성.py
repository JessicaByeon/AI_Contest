# 데이콘 문제풀기
# https://dacon.io/competitions/official/235961/overview/description

# 압축파일 생성

import os
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_pakchoi/test_target")
with zipfile.ZipFile("submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#1. 데이터
path = 'D:/study_data/_data/dacon_pakchoi/'
train_input_path = path + 'train_input/'
train_target_path = path + 'train_target/'
test_input_path = path + 'test_input/'
test_target_path = path + 'test_target/'

train_input_x01 = pd.read_csv(train_input_path+'CASE_01.csv')
print(train_input_x01.columns)
# Index(['시간', '내부온도관측치', '내부습도관측치', 'CO2관측치', 'EC관측치', '외부온도관측치', '외부습도관측치',
#        '펌프상태', '펌프작동남은시간', '최근분무량', '일간누적분무량', '냉방상태', '냉방작동남은시간', '난방상태',
#        '난방작동남은시간', '내부유동팬상태', '내부유동팬작동남은시간', '외부환기팬상태', '외부환기팬작동남은시간',
#        '화이트 LED상태', '화이트 LED작동남은시간', '화이트 LED동작강도', '레드 LED상태', '레드 LED작동남은시간',
#        '레드 LED동작강도', '블루 LED상태', '블루 LED작동남은시간', '블루 LED동작강도', '카메라상태', '냉방온도',
#        '난방온도', '기준온도', '난방부하', '냉방부하', '총추정광량', '백색광추정광량', '적색광추정광량',
#        '청색광추정광량'],
#       dtype='object')
# Index(['시간', '내부온도관측치', '내부습도관측치', 'CO2관측치', 'EC관측치', '외부온도관측치', '외부습도관측치',        
#        '펌프상태', '펌프작동남은시간', '최근분무량', '일간누적분무량', '냉방상태', '냉방작동남은시간', '난방상태',     
#        '난방작동남은시간', '내부유동팬상태', '내부유동팬작동남은시간', '외부환기팬상태', '외부환기팬작동남은시간',     
#        '화이트 LED상태', '화이트 LED작동남은시간', '화이트 LED동작강도', '레드 LED상태', '레드 LED작동남은시간',       
#        '레드 LED동작강도', '블루 LED상태', '블루 LED작동남은시간', '블루 LED동작강도', '카메라상태', '냉방온도',       
#        '난방온도', '기준온도', '난방부하', '냉방부하', '총추정광량', '백색광추정광량', '적색광추정광량',
#        '청색광추정광량', '외부온도추정관측치', '외부습도추정관측치', '펌프최근분무량', '펌프일간누적분무량'],
#       dtype='object')




# print(train_input.head)
# print(test_input.head)
# print(train_target.head)
# print(test_target.head)
print(train_input.columns)
print(test_input.columns)
print(train_target.columns)
print(test_target.columns)
print(train_input.info())
print(test_input.info())
print(train_target.info())
print(test_target.info())
print(train_input.describe())
print(test_input.describe())
print(train_target.describe())
print(test_target.describe())


print(np.array(train_input).shape) # (2653267, 43)
print(np.array(test_input).shape) # (335520, 42)
print(np.array(train_target).shape) # (1842, 2)
print(np.array(test_target).shape) # (230, 2)
print(train_input.isnull().sum()) # 시간을 제외한 모든 열에 결측치 존재
print(test_input.isnull().sum())

# # 결측치가 특히나 많은 열에 대한 decribe 확인
# print(train_input.describe())
# print(test_input.describe())
# print(train_target.describe())
# print(test_target.describe())

'''