# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:40:41 2023

@author: user
"""
#重跑時只需確認 Feeder(rets/cu_rets,noa) 和 Master 即可

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',None)

#母基金的挑選======================================================================
fund = pd.read_csv('Fund/data.csv', encoding='cp950')

#1.
fund[(fund['風險收益等級']=='RR2') & (fund['幣別']=='USD') &(fund['成立以來變動率(同類型排名)']==1.0)]
#T1034D 富全球投資債Bd
master = fund[fund['證券代碼']=='T1034D 富全球投資債Bd']

#2.取sharpe Ratio 在2017/09-2022/11最好的基金，且這段期間都要有資料(63筆月資料)，且要是美元計價
f1 = fund[(fund['年月日']>=20170901) & (fund['年月日']<=20221130) & (fund['幣別']=='USD')].loc[:,['年月日','基金全稱','淨值(元)','近一個月變動率%']]
f1 = f1.dropna()
f1['ym'] = f1['年月日']//100
f1['mR'] = f1['近一個月變動率%']*0.01
#取月底報酬率
f2 = f1.groupby(['基金全稱','ym']).apply(lambda f1 : f1.iloc[-1,:])
f2.index = np.arange(len(f2))
#剔除不滿足63筆的基金
number = f2.groupby('基金全稱')['ym'].count().reset_index() #4419
number = number[number['ym']==63] #1218
f3 = f2.merge(number['基金全稱'], on='基金全稱', how='inner')
#Sharpe Ratio
sr = f3.groupby('基金全稱').apply(lambda f3: (f3['mR'].mean()*12)/(f3['mR'].std()*np.sqrt(12))).reset_index().rename(columns={0:'Sharpe Ratio'})
print(sr[sr['Sharpe Ratio']==sr['Sharpe Ratio'].max()]) #兆豐美元貨幣市場證券投資信託基金
master = f3[f3['基金全稱']==sr[sr['Sharpe Ratio']==sr['Sharpe Ratio'].max()]['基金全稱'].values[0]]
master.insert(0,'Year',(f3['ym']//100).astype('int'))
master.insert(1,'Month',(f3['ym']%100).astype('int'))
master2 = master[['Year','Month','mR']]

#子基金的組成======================================================================================
rets = pd.read_csv('因子投組.csv').drop(['Unnamed: 0','cen','mom'],axis=1)
cu_rets = pd.read_csv('貨幣投組.csv').drop('Unnamed: 0',axis=1)
###多因子投組
noa = 3
w = pd.read_csv('Allocation敘統/多因子投組權重.csv').drop('Unnamed: 0',axis=1)
m = rets.merge(w,on='Date',how='inner')
feeder = m.groupby('Date').apply(lambda m:np.dot(m.iloc[:,1:noa+1],m.iloc[:,noa+1:-1].T)).reset_index().rename(columns={0:'dR'})
feeder['dR'] = feeder['dR'].astype('float')
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#2020-03只有一天，移除
feeder.drop(0,inplace=True)
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['dR'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})

###單因子投組(小市值)
feeder = rets[['Date','size']]
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['size'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})

###單因子投組(差流動性)
feeder = rets[['Date','liq']]
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['liq'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})

###單因子投組(高波動)
feeder = rets[['Date','vol']]
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['vol'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})

###單因子投組(高中心性)
feeder = rets[['Date','cen']]
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['cen'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})

###單因子投組(強動能)
feeder = rets[['Date','mom']]
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['mom'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})

###單一貨幣投組(BTC)
feeder = cu_rets[['Date','btc']]
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['btc'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})

###單一貨幣投組(ETH)
feeder = cu_rets[['Date','eth']]
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['eth'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})

###多貨幣投組
w = pd.read_csv('Allocation敘統/多貨幣投組權重.csv').drop('Unnamed: 0',axis=1)
m = cu_rets.merge(w,on='Date',how='inner')
feeder = m.groupby('Date').apply(lambda m:np.dot(m.iloc[:,1:3],m.iloc[:,3:5].T)).reset_index().rename(columns={0:'dR'})
feeder['dR'] = feeder['dR'].astype('float')
feeder['Year'] = pd.to_datetime(feeder['Date']).dt.year
feeder['Month'] = pd.to_datetime(feeder['Date']).dt.month
#計算及取得月底報酬率
feeder1 = feeder.groupby(['Year','Month']).apply(lambda feeder:feeder['dR'].cumsum().iloc[-1]).reset_index().rename(columns={0:'mR'})
#合併===================================================================================
All = master2.merge(feeder1,on=['Year','Month'],how='inner')
All.columns = ['Year','Month','MmR','FmR']
#分樣本內樣本外
#sa_in = All[All.index<=32]
#sa_out = All[All.index>32]

#母子基金運作機制======================================================================================================
#求停利贖回至母基金時間點
mflist = []
mlist = []
flist = []
#for Sample in [sa_in,sa_out]:
Sample = All
stoprofit = 1000 #不停利
a = Sample.iloc[0,3]
cumR = [a]
timelist = []
for i in range(1,len(Sample)):
    if a <stoprofit:
        #當期的累積報酬率 = (前一期的累積報酬率+1)*(當期的報酬率+1)-1
        a = (1+a)*(1+Sample.iloc[i,3])-1
        cumR.append(a)
    else:
        timelist.append(i)
        a = Sample.iloc[i,3]
        cumR.append(a)
Sample['cumR'] = cumR  

#模擬試算==================================================================
v0 = 100000
buy = 3000
f0 = buy
masterlist = [v0,v0*(1+Sample.iloc[0,2])-buy]
v0 = v0*(1+Sample.iloc[0,2])-buy
feederlist = [0,f0] 
bolist = []
for i in range(1,len(Sample)):
    if (i != len(Sample)-1) & (Sample.iloc[i,-1]<stoprofit) :
        #用當期的月初金額*(1+當期的月底報酬率)-下一期的申購金額，求得下一期的月初金額
        v1 = v0*(1+Sample.iloc[i,2])-buy
        masterlist.append(v1)
        v0 = v1
        #用當期的月初金額*(1+當期的月底報酬率)+下一期的定期定額，求得下一期的月初金額
        f1 = f0*(1+Sample.iloc[i,-2])+buy
        feederlist.append(f1)
        f0 = f1
    #停利贖回至母基金
    elif (i != len(Sample)-1) & (Sample.iloc[i,-1]>=stoprofit):
        f1 = f0*(1+Sample.iloc[i,-2])+buy - f0*(1+Sample.iloc[i,-2])
        feederlist.append(f1)
        bo = f0*(1+Sample.iloc[i,-2])
        bolist.append([i+1,bo])
        v1 = v0*(1+Sample.iloc[i,2])-buy+bo
        masterlist.append(v1)
        f0 = f1
        v0 = v1    
    else:
        v1 = v0*(1+Sample.iloc[i,2])
        masterlist.append(v1)
        f1 = f0*(1+Sample.iloc[i,-2])
        feederlist.append(f1)
        
port = pd.DataFrame(masterlist,columns=['Master'])
port.insert(0,'Date',list(Sample['Year']*100+Sample['Month'])+[202212])
port.insert(2,'Feeder',feederlist)
port['Total'] = port['Master']+port['Feeder']        
port['Return'] = port['Total'].pct_change()       
#MDD複利
port['cumR'] = (port['Return']+1).cumprod()-1  #單利 port['Return'].cumsum()
port['MDD'] = (1+port['cumR']).div((port['cumR']+1).cummax()).sub(1)
Sample['M_cumR'] = (Sample['MmR']+1).cumprod()-1 #單利 (Sample['MmR']).cumsum()
Sample['F_cumR'] = (Sample['FmR']+1).cumprod()-1 #單利 (Sample['FmR']).cumsum()
Sample['M_MDD'] = (1+Sample['M_cumR']).div((Sample['M_cumR']+1).cummax()).sub(1)
Sample['F_MDD'] = (1+Sample['F_cumR']).div((Sample['F_cumR']+1).cummax()).sub(1)

mflist = mflist + [(port['Return'].mean()*12)/(port['Return'].std()*np.sqrt(12)), port['cumR'].iloc[-1]*100, port['Return'].std()*np.sqrt(12)*100,abs(port['MDD'].min())*100,port.iloc[-1,-2]/abs(port['MDD'].min())]
mlist = mlist + [(Sample['MmR'].mean()*12)/(Sample['MmR'].std()*np.sqrt(12)), Sample['M_cumR'].iloc[-1]*100, Sample['MmR'].std()*np.sqrt(12)*100, abs(Sample['M_MDD'].min())*100, Sample.iloc[-1,5]/abs(Sample['M_MDD'].min())]
flist = flist + [(Sample['FmR'].mean()*12)/(Sample['FmR'].std()*np.sqrt(12)), Sample['F_cumR'].iloc[-1]*100, Sample['FmR'].std()*np.sqrt(12)*100, abs(Sample['F_MDD'].min())*100, Sample.iloc[-1,6]/abs(Sample['F_MDD'].min())]
  
pT = pd.DataFrame()
pT['Single-Currency Portfolio'] =['In Sample']*5 #+['Out Sample']*5 
pT['BTC'] = ['Sharpe Ratio','Total Return(%)','Risk(%)','MDD(%)','Return to MDD ratio']#*2
pT['Core-Satellite portfolio'] = mflist
pT['Core'] = mlist
pT['Satellite'] = flist  
pT.set_index(['Single-Currency Portfolio'],inplace=True)
pT
pT.to_csv('Fund/performance.csv',encoding='cp950')

#畫圖=========================================================================================
import matplotlib.pyplot as plt

port = port.dropna()
date = Sample['Year']*100+Sample['Month']
#Cumulative Return
plt.style.use('seaborn-ticks')
fig = plt.figure(figsize = (15,12))
ax1 = fig.add_subplot(111)
x = np.arange(len(date))
ax1.plot(x, Sample['M_cumR']*100, color='lightgreen', label='Core',linewidth=3)
plt.legend(loc=2,fontsize=14)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Cumulative Return rate(%)', fontsize=15)
plt.axhline(0,color="black",linestyle='--')
ax2 = ax1.twinx()
ax2.plot(x, port['cumR']*100, color='red',label='Core-Satellite portfolio',linewidth=3)
ax2.plot(x, Sample['F_cumR']*100, color='dodgerblue', label='Satellite', linewidth=3)
xlist = list(date)[::int(len(date)/10)]
xx = x[::int(len(date)/10)]
plt.xticks(xx, xlist)
ax1.tick_params(axis='both', labelsize=15)
ax2.tick_params(axis='both', labelsize=15)
plt.legend(loc=1,fontsize=14)
plt.axhline(0,color="black",linestyle='--')
plt.ylabel('Cumulative Return rate(%)', fontsize=15)
plt.title('Cumulative Returns among Core and Satellite portfolio', fontsize=18)
plt.savefig("Fund//Cumulative Returns.png")

#DD
plt.style.use('seaborn-ticks')
fig = plt.figure(figsize = (15,12))
ax1 = fig.add_subplot(111)
x = np.arange(len(date))
ax1.plot(x, Sample['M_MDD']*100, color='lightgreen', label='Core',linewidth=4)
plt.legend(loc=3,fontsize=14)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Drawdown(%)', fontsize=15)
plt.ylim(-110,5)
plt.axhline(0,color="black",linestyle='--')
ax2 = ax1.twinx()
ax2.plot(x, port['MDD']*100, color='red',label='Core-Satellite portfolio',linewidth=5)
ax2.plot(x, Sample['F_MDD']*100, color='dodgerblue', label='Satellite', linewidth=3)
xlist = list(date)[::int(len(date)/10)]
xx = x[::int(len(date)/10)]
plt.xticks(xx, xlist)
ax1.tick_params(axis='both', labelsize=15)
ax2.tick_params(axis='both', labelsize=15)
plt.legend(loc=4,fontsize=14)
plt.ylim(-110,5)
plt.ylabel('Drawdown(%)', fontsize=15)
plt.title('Drawdowns among Core and Satellite portfolio', fontsize=18)
plt.savefig("Fund//Drawdowns.png")
    
