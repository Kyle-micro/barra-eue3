# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
pd.set_option('display.max_columns',None)

data = pd.read_csv('DacsCrypto.csv').drop('Unnamed: 0',axis=1) 
dacs = pd.read_csv('DACS_data.csv').rename(columns={'Name':'name','Symbol':'symbol'}).drop('DACS Rank',axis=1)
estu = data.merge(dacs, on='symbol', how='inner')
estu = estu[['Date','symbol','name','Open','High','Low','Close','Volume','Market Cap','Sector','Industry Group','Industry']]
#ESTU
#市值標準，原本499種幣別，還是499種
estu = estu[estu['Market Cap']>=4500000]
len(estu['name'].unique())
#Listed and Trading標準，499剩491種
a = estu.groupby(['symbol'])['Date'].count().reset_index()
a.index = np.arange(a.shape[0])
a = a[a['Date']>=60]
estu = estu.merge(a['symbol'],on='symbol',how='inner')
#Liquidity - 60天交易量中位數不能低於45000元，491剩489種
estu = estu[estu['Volume']>0]
test = estu.groupby('name')['Volume'].rolling(60).median().reset_index()
test1 = test[test['Volume']>=45000]
test1.index = test1['level_1']
data1 = estu.merge(test1['level_1'], right_index=True, left_index=True).drop('level_1',axis=1)
data1 = data1.sort_values(by='Date',ignore_index=True)
len(data1['name'].unique())

#因子=====================================================================
#Size
data1['size'] = np.log(data1['Market Cap'])

#vol-DSTD
def halflife(a,b):
    lambd=(0.5**(1/a))
    lambdalist=[]
    for exp in range(b,-1,-1):
        lambdw=lambd**exp
        lambdalist.append(lambdw)
    lambdalist = np.array(lambdalist)/sum(lambdalist)
    return  lambdalist
#確認有足夠樣本數(65天以上)可以計算DSTD，489剩479種
number = data1.groupby('name')['Date'].count().reset_index().rename(columns={'Date':'number'})
data1 = data1.merge(number, on='name', how='inner')
data1 = data1[data1['number']>65]
len(data1['name'].unique())

data1['return'] = data1.groupby('name')['Close'].pct_change()
a = data1.groupby('name')['return'].mean().reset_index().rename(columns={'return':'mean'})
data1 = data1.merge(a,on='name',how='inner')
data1['diff'] = (data1['return'] - data1['mean'])**2
a=0
data1['volatility'] = np.nan
vol = pd.DataFrame()
for s in data1['name'].unique():
    s1 = data1[data1['name']==s].loc[:,['diff','volatility']]
    DSTD=[]
    for i in range(s1.shape[0]-65):
        dstd = np.sqrt(sum(s1.iloc[i+1:i+66,0]*halflife(23,64))/sum(halflife(23,64)))
        DSTD.append(dstd)
    s1.iloc[65:,1] = DSTD  
    vol = pd.concat([vol,s1])
    a+=1
    print(a)
data1['volatility'] = vol['volatility']

#Liq-Amihud illiquidity
data1['liquidity'] = np.log(1+ abs(data1['return'])/data1['Volume'])*(-1)
#data1['liq'] = abs(data1['return'])/data1['Volume']

#Mom
rf = pd.read_csv('3-Month Treasury Bill Secondary Market Rate, Discount Basis.csv')
rf['rf'] = np.where(rf['rf']=='.',0,rf['rf']).astype(float)
rf['rf'] = rf['rf']*0.01/252   
rf['Date'] = pd.to_datetime(rf['Date']).dt.date
data1['Date'] = pd.to_datetime(data1['Date']).dt.date
rf = rf[(rf['Date']>=data1['Date'].unique()[0]) & (rf['Date']<=data1['Date'].unique()[-1])]    
#填無風險利率缺值(周末和國定假日沒有)
dayT = pd.DataFrame(data1['Date'].unique()).rename(columns={0:'Date'})
dayt = dayT.merge(rf,on='Date',how='outer')   
for i in dayt[dayt['rf'].isna()].index:
    dayt.loc[i,'rf'] = dayt.loc[i-1,'rf']
data1 = data1.merge(dayt,on='Date',how='inner')   
data1['ln(1+r)'] = np.log(data1['return']+1)
data1['ln(1+rf)'] = np.log(data1['rf']+1)

a = data1.groupby('name')['ln(1+r)'].rolling(21).sum().reset_index()
b = data1.groupby('name')['ln(1+rf)'].rolling(21).sum().reset_index().drop('level_1',axis=1)
a = pd.concat([a,b['ln(1+rf)']],axis=1)
a['Momentum'] = a['ln(1+r)']-a['ln(1+rf)']
a.index = a['level_1']  
data1 = pd.merge(data1,a['Momentum'],right_index=True,left_index=True)
#動能要延遲一週
data1['lag_mom'] = data1.groupby('name')['Momentum'].shift(7)

#data1.to_csv('network_data.csv')
#Centrality
import networkx as nx
ret = data1.pivot(index='Date', columns='name',values='return')
degT = pd.DataFrame()
for i in range(ret.shape[0]-30+1):
    c = ret.iloc[i:i+30].dropna(axis=1).corr()
    c = c[(c>0.5) & (c!=1)]
    c.fillna(0, inplace=True)
    G = nx.Graph(c)
    deg_centrality = nx.degree_centrality(G)
    degT = degT.append([deg_centrality])
    print(i)
degT.insert(0,'Date',ret.index[29:])
degT1 = degT.melt(id_vars=['Date'], value_vars=degT.columns[1:], var_name='name', value_name='centrality').dropna()
data1 = data1.merge(degT1, on=['Date','name'], how='inner')

#求市值加權權重
sumcapT = data1.groupby('Date')['Market Cap'].sum().reset_index().rename(columns={'Market Cap':'sumcap'}) 
data1 = data1.merge(sumcapT, on='Date', how='inner') 
data1['cap_weighted'] = data1['Market Cap']/data1['sumcap']

#data1.to_csv('dacs_estu.csv')

factor = data1.loc[:,['Date','symbol','name','Sector','cap_weighted','size','volatility','liquidity','lag_mom','centrality']]
factor = factor.dropna()
factor.rename(columns={'lag_mom':'momentum'},inplace=True)
factor.to_csv('dacs_factor.csv')
#相關係數測一下
factor.iloc[:,5:].corr()

#===========================因子標準化 + 壓縮========================================================
f = pd.read_csv('dacs_factor.csv').drop('Unnamed: 0', axis=1)

#分佈圖
import seaborn as sns
sns.set() #設定格線
sns.displot(f['size'],kde=True)
plt.savefig('因子分佈圖/size.png')
sns.displot(f['volatility'],kde=True)
plt.savefig('因子分佈圖/volatility.png')
sns.displot(f['liquidity'],kde=True)
plt.savefig('因子分佈圖/liquidity.png')
sns.displot(f['momentum'],kde=True)
plt.savefig('因子分佈圖/momentum.png')
sns.displot(f['centrality'],kde=True)
plt.savefig('因子分佈圖/centrality.png')

#標準化
f1 = f.melt(id_vars=['Date', 'symbol', 'name','Sector','cap_weighted'], value_vars=f.columns[5:], var_name='factor')
f1['權重*值'] = f1['cap_weighted']*f1['value']
f2 = f1.groupby(['Date','factor'])['權重*值'].sum().reset_index().rename(columns={'權重*值':'mean'}) 
b = f1.groupby(['Date','factor'])['value'].std().reset_index().rename(columns={'value':'std'}) 
f2 = f2.merge(b, on=['Date','factor'], how='inner')
f1 = f1.merge(f2 ,on=['Date','factor'], how='inner')
f1['z'] = (f1['value']-f1['mean'])/f1['std']

f3 = f1
#水平推壓，只要max(z)沒有超過3.5或min(z)沒有小於-3.5，就沒有用
z = f3.groupby(['Date','factor'])['z'].max().reset_index().rename(columns={'z':'max'})
z1 = f3.groupby(['Date','factor'])['z'].min().reset_index().rename(columns={'z':'min'})
z = z.merge(z1, on=['Date','factor'], how='inner')

z['s_pos'] = [np.max([0,np.min([1,0.5/(i-3)])]) for i in z['max']]
z['s_neg'] = [np.max([0,np.min([1,(-0.5)/(i+3)])]) for i in z['min']]

f3 = f3.merge(z.drop(['max','min'],axis=1),on=['Date','factor'], how='inner')
f3['z1'] = np.where(f3['z']>3, 3*(1-f3['s_pos'])+f3['z']*f3['s_pos'], f3['z'])
f3['z1'] = np.where(f3['z1']<-3, -3*(1-f3['s_neg'])+f3['z1']*f3['s_neg'], f3['z1'])
f3['z1'] = np.where(f3['z1']==float('inf'), 3.5, f3['z1'])
f3['z1'] = np.where(f3['z1']==float('-inf'), -3.5, f3['z1'])

#水平推壓前後比較圖
test = pd.DataFrame()
test['aft'] = f3['z1']
test['bf'] = f3['z']
test['factor'] = f3['factor']
a = test[test['factor']=='size']
b = test[test['factor']=='liquidity']
c = test[test['factor']=='centrality']
d = test[test['factor']=='volatility']
e = test[test['factor']=='momentum']

plt.style.use('seaborn')
plt.figure(figsize = (15,12))
#plt.scatter(e['bf'], e['aft'], color='pink', label='momentum', s=30)
#plt.scatter(a['bf'], a['aft'], color='red', label='size',s=30)
#plt.scatter(b['bf'], b['aft'], color='green', label='liquidity',s=30)
plt.scatter(d['bf'], d['aft'], color='blue', label='volatility',s=30)
#plt.scatter(c['bf'], c['aft'], color='gold', label='centrality',s=30)
plt.xlabel('Before', fontsize=14)
plt.ylabel('After', fontsize=14)
plt.text(3.425, -4, 3.5, fontsize=12)
plt.text(-3.6, -4, -3.5, fontsize=12)
plt.text(-4.2, -3.55, -3.5, fontsize=12)
plt.text(-4.15, 3.48, 3.5, fontsize=12)
plt.axhline(3.5,ls='--')
plt.axhline(-3.5,ls='--')
plt.axvline(-3.5,ls='--')
plt.axvline(3.5,ls='--')
plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left',  fontsize=14)
plt.title('Before & After truncation comparison of Volatility', fontsize=18)
plt.savefig('因子分佈圖//cmc_volatility_Before & After truncation comparison.png')

#檢查f4有沒有跟f的列數相同
f4 = f3.pivot(index=list(f3.columns[:5]), columns='factor', values='z1').reset_index()
f5 = f4.dropna()
f4.shape[0]-f5.shape[0] 

#檢查因子有沒有共線性問題   
def checkVIF_new(df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    df['c'] = 1
    name = df.columns
    x = np.matrix(df)
    VIF_list = [variance_inflation_factor(x,i) for i in range(x.shape[1])]
    VIF = pd.DataFrame({'feature':name,"VIF":VIF_list})
    VIF = VIF.iloc[:-1]
    if len(np.where(VIF['VIF']>10)[0])!=0:
        print(date)  
for date in np.unique(f5['Date']):
    factorT5=f5[f5['Date']==date]
    VIFtest=factorT5[factorT5.columns[5:]]
    checkVIF_new(VIFtest) 
#'2017-05-21'後才沒有共線性
f6 = f5[f5['Date']>'2017-05-21']
#'2017-07-17'後才有一致的產業因子
number = []
for i in np.sort(f6['Date'].unique()):
    df = f6[f6['Date']==i]
    #ig = len(df['Industry Group'].unique())
    #ind = len(df['Industry'].unique())  
    s = len(df['Sector'].unique())  
    number.append([i,s])
Num = pd.DataFrame(number,columns=['Date','Sector'])
Num[Num['Sector']==6] #2017-07-17到2022-11-27有1960天
f6 = f5[f5['Date']>='2017-07-17']
f6.to_csv('dacs_factor_done.csv')

#平均相關係數(因子曝險)=================================================================
data = f6
dataXlist=[]
for date in np.unique(data['Date']):
    data1=data[data['Date']==date]
    data1['value1']=[1]*data1.shape[0]
    dataX = data1.pivot(index=['symbol']+list(data1.columns[5:10]), columns='Sector', values='value1').reset_index()
    dataX = dataX.merge(data1[['symbol','Sector']], on='symbol', how='inner')
    dataX=dataX.sort_values(by='Sector')
    dataX.fillna(0,inplace=True)
    dataX=dataX.drop(['symbol','Sector'],axis=1)
    corr=dataX.corr().iloc[:,:5].values
    dataXlist.append(corr)
    print(date)
    
cm1list=[]
for j in range(dataX.shape[1]):
        corr_mean=[]
        for i in range(len(dataXlist)):
            corr_mean.append(dataXlist[i][j])
        cm=pd.DataFrame(corr_mean)
        cm1=cm.describe().loc['mean'].values
        cm1list.append(cm1)    
corrT=pd.DataFrame(cm1list, columns=data.columns[5:], index=dataX.columns.values)
corrT.to_csv('因子曝險CorrT.csv')
