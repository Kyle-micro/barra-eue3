# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os 
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('dacs_factor_done.csv').drop('Unnamed: 0',axis=1)
df = df.sort_values(['Sector','name'])
#sqrt_cap_weighted
cap = pd.read_csv('dacs_estu.csv').drop('Unnamed: 0',axis=1)
cap['sqrt_cap'] = np.sqrt(cap['Market Cap'])
cap['cap_weighted_sqrt'] = cap.groupby('Date').apply(lambda cap:cap['sqrt_cap']/sum(cap['sqrt_cap'])).values
#Excesss Return
cap['excess return'] = cap['return']-cap['rf']

df = df.merge(cap[['Date','name','cap_weighted_sqrt','excess return']], on=['Date','name'],how='inner')
#計算因子報酬時，Y 要比 X 領先一期
df['leading_excess return'] = df.groupby('name')['excess return'].shift(-1) 
#拿掉每個幣別的最後一天
df = df.dropna()
#COCOS 2021/01/20 報酬1000多倍太恐怖拿掉
ii = df[(df['symbol']=='COCOS') & (df['Date']=='2021-01-19')].index[0]
df = df[df.index!=ii]
df.index = np.arange(df.shape[0])
#df.isna().sum()

fac_re = []
res = []
for i in np.sort(df['Date'].unique()):
    a = df[df['Date']==i]
       
    V = np.zeros([len(a),len(a)])
    np.fill_diagonal(V,a['cap_weighted_sqrt'])
    
    X = np.array(pd.concat([pd.get_dummies(a['Sector']),a.loc[:,a.columns[5:10]]],axis=1))
    X = np.insert(X,0,1,axis=1)
    
    R = np.zeros([X.shape[1]-1,X.shape[1]-1])
    np.fill_diagonal(R,1)
    GA = np.array(-a.groupby(['Sector']).sum()['cap_weighted_sqrt']/a.groupby(['Sector']).sum()['cap_weighted_sqrt'][-1])[:-1]
    GA = np.append(np.append([0],GA),[0,0,0,0,0])
    R = np.insert(R,X.shape[1]-6,GA,axis=0)
    
    inv = np.dot(np.dot(np.dot(np.dot(R.T,X.T),V),X),R)
    inv = np.linalg.inv(inv)
    omega = np.dot(np.dot(np.dot(np.dot(R,inv),R.T),X.T),V)
    
    fac = np.dot(omega,np.array(a['leading_excess return']))
    fac_re.append(fac)
    res.append((np.array(a['leading_excess return']) - np.dot(X,fac)))
    print(i)

#R-square
rsqrlist = []
res = pd.DataFrame(res)    
k=0
for i in np.sort(df['Date'].unique()):
    a = df[df['Date']==i]
    wei = np.array(a['cap_weighted_sqrt'])
    top = np.array(pow(res.iloc[k,:len(a)],2)).dot(wei)
    down = np.array(pow(a['leading_excess return'],2)).dot(wei)
    rsqr = 1-(top/down)
    rsqrlist.append([i,rsqr])
    k+=1
    print(i)

rsqrlistT = pd.DataFrame(rsqrlist)
rsqrlistT.describe().to_csv('因子報酬敘述統計/R2敘述統計.csv')
rsqrlistT.to_csv('因子報酬敘述統計/R-square_adj.csv') 

#jump
res_1 = 1.4826 * ((abs(res - res.median(axis=1))).median(axis=1))   # sigma_u
res_ = pd.DataFrame()
for i in range(0,len(res.columns)):
    res_ = pd.concat((res_,res_1),axis=1)
res_.columns = res.columns
res_2 = (res[abs(res) > (4 * res_)] - (4 * res_)) # 若res的絕對值大於4倍 sigma_u ， 則 res - 4*sigma_u 再乘以 res 的正負號。
res_3 = (np.sign(res)*res_2) . fillna(0) #把有值的乘以原本的正負號，剩下的擺 0
 
excess = []   
for i in np.sort(df['Date'].unique()):
    a = df[df['Date']==i]
    excess.append(np.array(a['leading_excess return']))
    print(i)
excess = pd.DataFrame(excess)      

new_excess = excess - res_3

#============================第二次求解==========================================
k=0
new_fac = []
#new_res = []
for i in np.sort(df['Date'].unique()):
    a = df[df['Date']==i]
    
    V = np.zeros([len(a),len(a)])
    np.fill_diagonal(V,a['cap_weighted_sqrt'])
    
    X = np.array(pd.concat([pd.get_dummies(a['Sector']),a.loc[:,a.columns[5:10]]],axis=1))
    X = np.insert(X,0,1,axis=1)
    
    R = np.zeros([X.shape[1]-1,X.shape[1]-1])
    np.fill_diagonal(R,1)
    GA = np.array(-a.groupby(['Sector']).sum()['cap_weighted_sqrt']/a.groupby(['Sector']).sum()['cap_weighted_sqrt'][-1])[:-1]
    GA = np.append(np.append([0],GA),[0,0,0,0,0])
    R = np.insert(R,X.shape[1]-6,GA,axis=0)
    
    inv = np.dot(np.dot(np.dot(np.dot(R.T,X.T),V),X),R)
    inv = np.linalg.inv(inv)
    omega = np.dot(np.dot(np.dot(np.dot(R,inv),R.T),X.T),V)
    
    fac = np.dot(omega,np.array(new_excess.iloc[k,:omega.shape[1]]))
    #new_res.append((new_excess.iloc[k,:omega.shape[1]] - np.dot(X,fac)))
    new_fac.append(fac)
    k+=1
    print(i)

#創建因子報酬表
new_fac = pd.DataFrame(new_fac) 
#new_res = pd.DataFrame(new_res)
new_fac.index = np.sort(df['Date'].unique())
new_fac.columns = ['Market'] + list(np.sort(df['Sector'].unique())) + list(a.columns[5:10])
#new_res.to_csv('factor_res.csv')
new_fac.insert(0,'Date',np.sort(df['Date'].unique()))
new_fac.to_csv('因子報酬敘述統計/factor_return_adj.csv')


