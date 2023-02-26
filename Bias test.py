# 重跑要設Sample, o, 

import pandas as pd
import numpy as np
import statsmodels.api as sm

#投組標準差
#Common Factor Risk
fac_ret = pd.read_csv('因子報酬敘述統計/factor_return_adj.csv').set_index('Unnamed: 0')
sam_size=540
#sam_size=1500
n = fac_ret.shape[0]-sam_size #短 長'2019-01-08'
Cmlist=[]     
for i in range(n):
    a1=np.load('Common Factor Risk_Long\Cm_'+str(i)+'.npy')
    Cmlist.append(a1)

#讀因子loading處理好的資料
data = pd.read_csv('dacs_factor_done.csv').drop(columns=['Unnamed: 0'])
#排除 COCOS 當時沒有用到的loading
ii = data[(data['symbol']=='COCOS') & (data['Date']=='2021-01-19')].index[0]
data = data[data.index!=ii]
#長期
data = data[data['Date'] >= data['Date'].unique()[sam_size]]
#短期
#data = data[data['Date']>='2016-02-04']
ret = pd.read_csv('dacs_estu.csv').drop('Unnamed: 0',axis=1)
data = data.merge(ret[['Date','name','return']], on=['Date','name'], how='inner')
data['yearmon'] =  pd.to_datetime(data['Date']).dt.strftime('%Y%m') 
data = data.sort_values(['Sector','name'])

#風格因子投組--每個月底的曝險值決定下個月的幣別
for i in ['lag_'+i for i in list(data.columns[5:10])]:
    data[i] = data.groupby('symbol')[i[4:]].shift()
n1 = 30
data['Day'] = pd.to_datetime(data['Date']).dt.day
#小市值
sizeP = data.groupby(['yearmon']).apply(lambda data : data[data['Day']==1].sort_values(by='lag_size').iloc[:n1])
sizeP.index = np.arange(sizeP.shape[0])
sizeP1 = data.merge(sizeP[['symbol','yearmon']], on=['symbol','yearmon'], how='inner')
sizeP1 = sizeP1.sort_values('Date')
change_w1 = sizeP1.groupby('Date').apply(lambda sizeP1: sizeP1['cap_weighted']/sum(sizeP1['cap_weighted'])).reset_index()
sizeP1['cap_weighted'] = change_w1['cap_weighted'].values

Sample = data
Sample.index = np.arange(len(Sample))
#市值加權投組風險============================================================================
sdev_plist=[]
o = 0 #開始日從2019-01-08
#o = len(data[data['Date']< '2019-02-01']['Date'].unique()) #開始日從2019-02-01
#[:-1]是因為因子曝險比因子報酬多一期，而這又是因為在計算因子報酬時， Y 要比 X 提前一期
for date in np.sort(np.unique(Sample['Date']))[:-1]:
    data1 = Sample[Sample['Date']==date]
    
    # X 因子曝險
    X = np.array(pd.concat([pd.get_dummies(data1['Sector']),data1.loc[:,data1.columns[5:10]]],axis=1))
    X = np.insert(X,0,1,axis=1)
    
    X_P = np.array(data1['cap_weighted']).T.dot(X)
    sdev_p=np.sqrt(np.dot(X_P,Cmlist[o]).dot(X_P.T))
    #獨有風險 np.dot(np.array(pow(data1['cap_weighted'],2)),np.array(data1['Specific_Risk']).T)
    sdev_plist.append([date, sdev_p])
    o+=1
    print(date)

MT = pd.DataFrame(sdev_plist, columns=['date', 'sdev_p'])
#short_sdev_pT=pd.DataFrame(sdev_plist, columns=['date', 'sdev_p'])

#等權重投組風險==================================================================================
sdev_pElist=[]
o=0
for date in np.sort(np.unique(data['Date']))[:-1]:
    data1 = Sample[Sample['Date']==date]
    
    # X 因子曝險
    X = np.array(pd.concat([pd.get_dummies(data1['Sector']),data1.loc[:,data1.columns[5:10]]],axis=1))
    X = np.insert(X,0,1,axis=1)
    
    X_P = sum(X)/data1.shape[0]
    sdev_p = np.sqrt(np.dot(X_P,Cmlist[o]).dot(X_P.T))
    #獨有風險 sum(data1['Specific_Risk']/pow(data1.shape[0],2))
    sdev_pElist.append([date, sdev_p])
    o+=1
    print(date)

ET = pd.DataFrame(sdev_pElist, columns=['date', 'sdev_p'])
#short_sdev_pET=pd.DataFrame(sdev_pElist, columns=['date', 'sdev_p'])

#Monthly return==================================================================================
#排除COCOS的誇張報酬
data = data[data['return']!=data[data['symbol']=='COCOS']['return'].max()]
cap = data.groupby('Date').apply(lambda data: sum(data['return'] * data['cap_weighted'])).reset_index().rename(columns={0:'dr'})
equ = data.groupby('Date')['return'].mean().reset_index().rename(columns={'return':'dr'})
cap['yearmon'] = pd.to_datetime(cap['Date']).dt.strftime('%Y%m')
capm = cap.groupby('yearmon').apply(lambda cap: (cap['dr']+1).prod()-1).reset_index().rename(columns={0:'mr'})
equ['yearmon'] = pd.to_datetime(equ['Date']).dt.strftime('%Y%m')
equm = equ.groupby('yearmon').apply(lambda cap: (cap['dr']+1).prod()-1).reset_index().rename(columns={0:'mr'})

'''
date_ = np.sort(data['Date'].unique())[:-1]
port_return = []
for i in date_:
    a = ret[ret['Date']==i]
    equal = sum(a['return'] * (1/len(a)))
    cap = sum(a['return'] * a['cap_weighted'])
    port_return.append([i,equal,cap])
    print(i) 
port = pd.DataFrame(port_return,columns=['date','equal','cap'])
port['date'] = pd.to_datetime(port['date']).dt.strftime('%Y%m')
port = port.set_index('date')
port = port + 1
port_mon = port.groupby('date').prod()-1
'''
##小市值
#daily return of size portfolio
sizeP2 = sizeP1.groupby('Date').apply(lambda sizeP1: sum(sizeP1['cap_weighted']*sizeP1['return'])).reset_index().rename(columns={0:'dr'}) 
sizeP2['yearmon'] = pd.to_datetime(sizeP2['Date']).dt.strftime('%Y%m')
#monthly return of size portfolio
ms = sizeP2.groupby('yearmon').apply(lambda sizeP2: (sizeP2['dr']+1).prod()-1).reset_index().rename(columns={0:'mr'})


#=================Bias test 用月初的投組風險看這個月的報酬======================================================================= 
#每天的 Monthly risk of portfolio
drp = MT
#monthly return of portfolio
mp = capm

#月初的 Monthly risk of portfolio
drp['yearmon'] = pd.to_datetime(drp['date']).dt.strftime('%Y%m')
drp['day'] = pd.to_datetime(drp['date']).dt.day
mrp = drp.groupby('yearmon').apply(lambda drp: drp.iloc[0,:])
z = pd.DataFrame(mp['mr'].values/mrp['sdev_p'].values)
z_sd = z.rolling(12).std()[11:] 

#short_ze_std = ze.rolling(12).std()[11:]   

bias = [ i for i in z_sd.values if i < 1+np.sqrt(1/6) and i>1-np.sqrt(1/6)]
ratio = len(bias)/ z_sd.shape[0]
print(ratio) #S:；L:47.22%

#Robust
zR = []
for i in np.array(mp['mr'])/np.array(mrp['sdev_p']):
    zR.append(max(-3,min(3,i)))
zR = pd.DataFrame(zR)
zR_sd = zR.rolling(12).std()[11:] 
 
#short_zeR_std = zeR.rolling(12).std()[11:]  

biasR = [ i for i in zR_sd.values if i < 1+np.sqrt(1/6) and i>1-np.sqrt(1/6)]
ratio = len(biasR)/zR_sd.shape[0]
print(ratio) #S:；L:50.00%

#市值加權=======================================================================    
#月初投組風險
sdev_pT['month'] = pd.to_datetime(sdev_pT['date']).dt.strftime('%Y%m')
std_p = []
for i in sdev_pT['month'].unique():
    std_p.append(sdev_pT[sdev_pT['month'] == i]['sdev_p'].iloc[0])
z = np.array(port_mon['cap'])/np.array(std_p)
Z=pd.DataFrame(z)
z_std = Z.rolling(12).std()[11:]
 
#short_z_std = Z.rolling(12).std()[11:]    
    
test=[ i for i in z_std.values if i < 1+np.sqrt(1/6) and i>1-np.sqrt(1/6)]
ratio = len(test)/ z_std.shape[0]
print(ratio) #S:；L:63.88%

#Robust
zR = []
for i in np.array(port_mon['cap'])/np.array(std_p):
    zR.append(max(-3,min(3,i)))
zR = pd.DataFrame(zR)
zR_std = zR.rolling(12).std()[11:] 

#short_zR_std = zR.rolling(12).std()[11:] 

testR=[ i for i in zR_std.values if i < 1+np.sqrt(1/6) and i>1-np.sqrt(1/6)]
ratio = len(testR)/ zR_std.shape[0]
print(ratio) #S:；L:88.89%

#-------------------------------------圖---------------------------------------
#無Robust
##市值:L+S
z_std.insert(0,'date',sdev_pT['month'].unique()[11:])
##等權:L+S
ze_std.insert(0,'date',sdev_pT['month'].unique()[11:])


#有Robust
##市值:L+S
zR_std.insert(0,'date',sdev_pT['month'].unique()[11:])
##等權:L+S
zeR_std.insert(0,'date',sdev_pT['month'].unique()[11:])

#市值加權
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.figure(figsize = (12,8)).subplots_adjust(top=0.8)
x = np.arange(len(z_std))
plt.scatter( x, z_std[0], color='red' ,label='Long Term')
#plt.scatter( Cap['date'], Cap['0_y'], color='gold' ,label='Long Term' ,marker='^')
plt.axhline(1+np.sqrt(1/6),color="blue",linestyle='--')
plt.axhline(1-np.sqrt(1/6),color="blue",linestyle='--')
plt.axhline(1,color="black",linestyle='--')
xlist = list(z_std['date'])[::int(len(z_std['date'])/8-1)]
xx = list(x[::int(len(z_std['date'])/8-1)])
plt.xticks(xx, xlist,fontsize=13, rotation=30)
plt.yticks(fontsize=13)
plt.ylim(0.5,1.7) 
plt.legend(loc=1)
plt.xlabel('Date', fontsize=13)
plt.ylabel('b value', fontsize=13)
plt.title('Cap Weight Bias Test', fontsize=16)
plt.savefig('Bias test/adj_Cap Weight Bias Test.png')

#市值加權有Robust
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.figure(figsize = (12,8)).subplots_adjust(top=0.8)
x = np.arange(len(z_std))
plt.scatter( x, zR_std[0], color='red' ,label='Long Term')
#plt.scatter( Cap['date'], Cap['0_y'], color='gold' ,label='Long Term' ,marker='^')
plt.axhline(1+np.sqrt(1/6),color="blue",linestyle='--')
plt.axhline(1-np.sqrt(1/6),color="blue",linestyle='--')
plt.axhline(1,color="black",linestyle='--')
xlist = list(z_std['date'])[::int(len(z_std['date'])/8-1)]
xx = list(x[::int(len(z_std['date'])/8-1)])
plt.xticks(xx, xlist,fontsize=13, rotation=30)
plt.yticks(fontsize=13)
plt.ylim(0.5,1.7)
plt.legend(loc=1)
plt.xlabel('Date', fontsize=13)
plt.ylabel('b value', fontsize=13)
plt.title('Cap Weight Robust Bias Test', fontsize=16)
plt.savefig('Bias test/adj_Cap Weight Robust Bias Test.png')

#等權無Robust
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.figure(figsize = (12,8)).subplots_adjust(top=0.8)
x = np.arange(len(ze_std))
plt.scatter( x, ze_std[0], color='red' ,label='Long Term')
#plt.scatter( Cap['date'], Cap['0_y'], color='gold' ,label='Long Term' ,marker='^')
plt.axhline(1+np.sqrt(1/6),color="blue",linestyle='--')
plt.axhline(1-np.sqrt(1/6),color="blue",linestyle='--')
plt.axhline(1,color="black",linestyle='--')
xlist = list(ze_std['date'])[::int(len(ze_std['date'])/8-1)]
xx = list(x[::int(len(ze_std['date'])/8-1)])
plt.xticks(xx, xlist,fontsize=13, rotation=30)
plt.yticks(fontsize=13)
plt.ylim(0.5,3)
plt.legend(loc=1)
plt.xlabel('Date', fontsize=13)
plt.ylabel('b value', fontsize=13)
plt.title('Equal Weight Bias Test', fontsize=16)
plt.savefig('Bias test/adj_Equal Weight Bias Test.png')

#等權有Robust
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.figure(figsize = (12,8)).subplots_adjust(top=0.8)
x = np.arange(len(zeR_std))
plt.scatter( x, zeR_std[0], color='red' ,label='Long Term')
#plt.scatter( Cap['date'], Cap['0_y'], color='gold' ,label='Long Term' ,marker='^')
plt.axhline(1+np.sqrt(1/6),color="blue",linestyle='--')
plt.axhline(1-np.sqrt(1/6),color="blue",linestyle='--')
plt.axhline(1,color="black",linestyle='--')
xlist = list(z_std['date'])[::int(len(z_std['date'])/8-1)]
xx = list(x[::int(len(z_std['date'])/8-1)])
plt.xticks(xx, xlist,fontsize=13, rotation=30)
plt.yticks(fontsize=13)
plt.ylim(0.5,3)
plt.legend(loc=1)
plt.xlabel('Date', fontsize=13)
plt.ylabel('b value', fontsize=13)
plt.title('Equal Weight Robust Bias Test', fontsize=16)
plt.savefig('Bias test/adj_Equal Weight Robust Bias Test.png')

#RAD------------------------------------------------------
#無Robust
abs(ze_std[0]-1).mean() #0.7966
#abs(Equal['0_y'].dropna()-1).mean() #0.6111
abs(z_std[0]-1).mean() #0.3446
#abs(Cap['0_y'].dropna()-1).mean() #0.5552

#Robust
abs(zeR_std[0]-1).mean() #0.3727
#abs(Equal['0_y'].dropna()-1).mean() #0.3999
abs(zR_std[0]-1).mean() #0.2548
#abs(Cap['0_y'].dropna()-1).mean() #0.3995
















