#重跑時只需確認 rets,noa 

import math
import numpy as np
import pandas as pd
from pylab import plt
np.set_printoptions(suppress=True)
from scipy.optimize import minimize

def portfolio_return(weights, rets):
    return np.dot(weights.T, rets.mean()) * 252

def portfolio_variance(weights, rets):
    return np.dot(weights.T, np.dot(rets.cov(), weights)) * 252

def portfolio_volatility(weights, rets):
    return math.sqrt(portfolio_variance(weights, rets))
#定義相對的風險歸因函數
def rel_risk_contributions(weights, rets):
    vol = portfolio_volatility(weights, rets)
    cov = rets.cov()
    mvols = np.dot(cov, weights) / vol
    rc = mvols * weights
    rrc = rc / rc.sum()
    return rrc

noa = 3 #標的資產數量
weights =  np.array(noa * [1 / noa]) #等權重
rets = pd.read_csv('因子投組.csv').drop(['Unnamed: 0','cen','mom'],axis=1)
n=30 #以30天的報酬率計算投組風險，決定下一天的權重
rets1 = rets.iloc[:n,1:] 
'''
rrc = rel_risk_contributions(weights,rets1)
plt.pie(rrc, labels=rets1.columns, autopct='%1.1f%%')
plt.title('Equally Weighted Portfolio')
portfolio_volatility(weights, rets1) # 1.0324
'''
#最小化實際的rrc與目標的rrc
def mse_risk_contributions(weights, target, rets):
    rrc = rel_risk_contributions(weights, rets)
    mse = ((rrc - target) ** 2).mean()
    return mse * 100
#限制式 : 權重加總為1
cons = {'type': 'eq', 'fun': lambda weights: weights.sum() - 1}
#限制式 :權重介於0-1之間 
bnds = noa * [(0, 1),]
#目標的相對風險歸因
target = noa * [1 / noa,]
#求解符合目標下的標的資產權重
opt = minimize(lambda w: mse_risk_contributions(w, target=target,rets=rets1), weights, bounds=bnds, constraints=cons)
weights_ = opt['x']
'''
plt.pie(weights_, labels=rets1.columns, autopct='%1.1f%%')
plt.title('Optimal Portfolio Weights')
#檢查rrc有沒有符合目標  #畫圖
rrc = rel_risk_contributions(opt['x'],rets1)
plt.pie(rrc, labels=rets1.columns,labeldistance=1.05, autopct='%1.1f%%',textprops={'weight':'bold', 'size':12,'family' : 'serif'}, wedgeprops={'linewidth':3,'edgecolor':'w'})
plt.title('Relative Risk Contributions',size=16,family='serif')
plt.axis('equal')
plt.savefig('Allocation敘統/Risk_parity.png')
#plt.legend(loc=2)
portfolio_volatility(opt['x'], rets1) #0.9639
'''

#再不調整權重的情況下，新權重=原本權重*(1+報酬率)/sum(原本權重*(1+報酬率))
w0 =  weights_ #初始權重
dev = 0.15
up_bw = np.array(target)*(1+dev)
low_bw = np.array(target)*(1-dev)
wlist = [list(weights_)+['Risk Parity']]
for i in range(1,rets.shape[0]-n+1):
    #隨著時間經過資料筆數越來越多，不是移動窗格
    rets1 = rets.iloc[:n+i]
    w1 = np.array((w0*(1+rets.iloc[n+i-1,1:]))/sum(w0*(1+rets.iloc[n+i-1,1:])))
    rrc1 = rel_risk_contributions(w1,rets1)
    if sum(rrc1 > low_bw)+sum(rrc1 < up_bw)==2*noa:
        wlist.append(list(w1)+[0])
    else:
        opt = minimize(lambda w: mse_risk_contributions(w, target=target,rets=rets1), w0, bounds=bnds, constraints=cons)
        w1 = opt['x']
        wlist.append(list(w1)+[1])
    w0 = w1
    
wT = pd.DataFrame(wlist)
Date = list(rets['Date'][n:].values)+['2022-11-28']
wT.insert(0,'Date',Date)      
wT.columns = ['Date'] + list(rets.columns[1:].values) + ['Rebalancing']     
wT.to_csv('Allocation敘統/多因子投組權重.csv')

#比較
m = rets.merge(wT,on='Date',how='inner')
mfp = m.groupby('Date').apply(lambda m:np.dot(m.iloc[:,1:noa+1],m.iloc[:,noa+1:-1].T)).reset_index().rename(columns={0:'Multi-Factor Portfolio'})
mfp['Multi-Factor Portfolio'] = mfp['Multi-Factor Portfolio'].astype('float')
print(mfp.cumsum().iloc[-1,-1]*100) #'總報酬 '
print(mfp.describe().iloc[2,0]*np.sqrt(365)*100) #'年化標準差 '
print(wT[wT['Rebalancing']==1].shape[0]/wT.shape[0]*100)#Rebalancing 96次，占總天數1916天5% 

#圖=============================================================================
plt.figure(figsize = (15,12))
plt.stackplot(wT['Date'],wT['btc'],wT['eth'],labels=['BTC','ETH'],colors=['r','lightblue'])
plt.legend(bbox_to_anchor=(1.15, 1),fontsize=13)
plt.tight_layout()
plt.ylabel('Weight',fontsize=13)
plt.xlabel('Date',fontsize=13)
x = np.arange(len(wT))
plt.ylim(0,1)
plt.xlim(0,x[-1])
xx = x[::int(len(wT)/10)]
xlist = list(wT['Date'])[::int(len(wT)/10)]
plt.xticks(xx, xlist,fontsize=13, rotation=30)
plt.title('The asset allocation of multi-currency Portfolio',fontsize=18,family='serif')
plt.savefig('Allocation敘統/多貨幣投組權重.png')

plt.figure(figsize = (15,12))
plt.stackplot(wT['Date'],wT['size'],wT['liq'],wT['cen'],wT['mom'],labels=['Size','Liquidity','Centrality','Momentum'],colors=['r','gold','lightblue','green'])
plt.legend(bbox_to_anchor=(1.15, 1),fontsize=13)
plt.tight_layout()
plt.ylabel('Weight',fontsize=13)
plt.xlabel('Date',fontsize=13)
x = np.arange(len(wT))
plt.ylim(0,1)
plt.xlim(0,x[-1])
xx = x[::int(len(wT)/10)]
xlist = list(wT['Date'])[::int(len(wT)/10)]
plt.xticks(xx, xlist,fontsize=13, rotation=30)
plt.title('The asset allocation of multi-currency Portfolio',fontsize=18,family='serif')
plt.savefig('Allocation敘統/4因子投組權重.png')

#折線圖
plt.style.use('seaborn')
plt.figure(figsize = (15,12))
x = np.arange(len(wT))
plt.plot(x, wT['size'],label='Size')
plt.plot(x, wT['liq'],label='Liquidity')
plt.plot(x, wT['vol'],label='Volatility')
plt.legend(loc=1,fontsize=15)
plt.ylim(0.15,0.5)
plt.xlim(0,x[-1])
xx = x[::int(len(wT)/10)]
xlist = list(wT['Date'])[::int(len(wT)/10)]
plt.xticks(xx, xlist,fontsize=13, rotation=30)
plt.yticks( fontsize=15)
plt.ylabel('Weight',fontsize=15)
plt.xlabel('Date',fontsize=15)
plt.title('The asset allocation of multi-factor Portfolio' , fontsize=18)
plt.savefig('Allocation敘統/多因子投組權重_折線圖.png')
'''
#投組的報酬與風險================================================================
w = pd.read_csv('Allocation敘統/多因子投組權重.csv').drop('Unnamed: 0',axis=1)
m = rets.merge(w,on='Date',how='inner')
mfp = m.groupby('Date').apply(lambda m:np.dot(m.iloc[:,1:4],m.iloc[:,4:7].T)).reset_index().rename(columns={0:'Multi-Factor Portfolio'})
mfp['Multi-Factor Portfolio'] = mfp['Multi-Factor Portfolio'].astype('float')
All = mfp.merge(rets, on='Date', how='inner')


#bandwidth======================================================================
K = 0.05
C = 0.001*2




    
    
    