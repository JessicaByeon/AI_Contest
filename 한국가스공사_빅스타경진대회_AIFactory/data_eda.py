import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,RobustScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
path = 'D:\\ai_data\AI_FACTORY\kogas/'

data_a = pd.read_csv(path+'월별공급량및비중.csv')
data_b = pd.read_csv(path+'상업용 상대가격(기준=2015).csv')
data_c = pd.read_csv(path+'제조업 부가가치(분기별).csv')
data_d = pd.read_csv(path+'추가데이터(1).csv',thousands = ',')


data_c = data_c.rename(columns={'제조업부가가치액':'MFG'})
data_c = pd.concat([data_c,data_c,data_c],ignore_index=True)
data_c = data_c.sort_values(by=['YEAR', 'QUARTER'],axis=0,ascending=True)
idx = pd.date_range('01-01-1996','12-31-2020',freq='M')

data_c['date'] = pd.to_datetime(idx)
data_c['MONTH'] = data_c['date'].dt.month
data_c = data_c.drop(['date'],axis=1)
data_c = data_c.reset_index(drop=True)
data_c = data_c[['YEAR','MONTH','MFG','QUARTER']]

data_all = pd.merge(data_a,data_b)
data_all = data_all.merge(data_c)
data_all = data_all.merge(data_d)


data_all = pd.get_dummies(data_all,columns=['QUARTER'])
print(data_all.columns)
# Index(['YEAR', 'MONTH', '도시가스(톤)_민수용', '도시가스(톤)_산업용', '도시가스(톤)_총합(민수용+산업용)',
#        '민수용비중', '산업용비중', 'RP(상대가격)', 'GAS_PRICE(산업용도시가스)',
#        'OIL_PRICE(원유정제처리제품)', 'MFG', '평균기온', '난방도일', '냉방도일', '최저기온', '최고기온',
#        '천연가스생산량(백만 m₂)', '산업소비량(백만 m₂)', '가정소비량(백만 m₂)', 'QUARTER_Q1',
#        'QUARTER_Q2', 'QUARTER_Q3', 'QUARTER_Q4'],
#       dtype='object')
data_all = data_all.rename(
    columns={'도시가스(톤)_민수용':'CIVIL','도시가스(톤)_산업용':'IND',
             'RP(상대가격)':'RP','민수용비중':'CIVILper',
             '산업용비중':'INDper','GAS_PRICE(산업용도시가스)':'GAS_PRICE','OIL_PRICE(원유정제처리제품)':'OIL_PRICE',
             '평균기온':'Meantemp','난방도일':'nanbangdoil','냉방도일':'naengbangdoil',
             '최저기온':'lowtemp','최고기온':'hightemp','천연가스생산량(백만 m₂)':'amount_of_gas',
             '산업소비량(백만 m₂)':'INDcon','가정소비량(백만 m₂)':'CIVILcon','도시가스(톤)_총합(민수용+산업용)':'Total',}
)
# 전월비 증감량
data_all['MOM_CIVIL'] = data_all['CIVIL'].pct_change()
data_all['MOM_IND'] = data_all['IND'].pct_change()

data_all.to_csv(path+'Total_data.csv',index=False)
month_data = data_all['MONTH'].unique()

CIVIL_MEAN = []
IND_MEAN = []
for j in month_data:
    civil_month = data_all[['CIVIL']][(data_all['MONTH']==j)].reset_index(drop=True)
    ind_month = data_all[['IND']][(data_all['MONTH']==j)].reset_index(drop=True)
    CIVIL = []
    IND = []
    for i in range(len(civil_month)):
        if i+1 == len(civil_month):
            break
        else :
            civil2 = civil_month.loc[i+1] - civil_month.loc[i]
            ind2 = ind_month.loc[i+1] - ind_month.loc[i]
            CIVIL.append(civil2)
            IND.append(ind2)
    CIVIL = np.round(np.mean(CIVIL),0)
    CIVIL_MEAN.append(CIVIL)
    IND = np.round(np.mean(IND),0)
    IND_MEAN.append(IND)
CIVIL_MEAN = (np.array(CIVIL_MEAN)).reshape(-1,1)
IND_MEAN = (np.array(IND_MEAN)).reshape(-1,1)
total = np.concatenate((CIVIL_MEAN,IND_MEAN),1)
print(CIVIL_MEAN)
np.save(path+'mean_data.npy',arr=total)










