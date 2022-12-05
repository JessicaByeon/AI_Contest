#4. 평가, 예측 
results = cat.score(x_test, y_test)
print('score',results)

y_pred = cat.predict(x_test)

y123 = y_pred[-24:] 
# print('2021-01-12', y123.shape) # (831, 2) 
mean_data1 = np.concatenate([mean_data, mean_data]) 
idx = 6 
L = [] 
for i in range(idx):

    y12 = y123 + (mean_data1*i) 

    L.append(y12) 

L = np.array(L).reshape(-1,2) 
y123 = np.concatenate([y123, L])
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)