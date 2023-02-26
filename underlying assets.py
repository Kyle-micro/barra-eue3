# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)

data = pd.read_csv('dacs_factor_done.csv').drop('Unnamed: 0',axis=1)
ret = pd.read_csv('dacs_estu.csv').drop('Unnamed: 0',axis=1)
data = data.merge(ret[['Date','name','return']], on=['Date','name'], how='inner')
data = data.dropna()
#排除COCOS在2021-01-20大暴漲1000多倍(離群值)
dat = data[data.index!=153496]

#貨幣投組===============================================================================
btc = dat[dat['symbol'] == 'BTC'][['Date','return']]
eth = dat[dat['symbol'] == 'ETH'][['Date','return']]
cu = pd.DataFrame()
cu['Date'] = btc['Date'].values
cu['btc'] = btc['return'].values
cu['eth'] = eth['return'].values
cu.to_csv('貨幣投組.csv')

#風格因子投組：小市值、差流動性、高波動、高中心性等==============================================
#確保投組由30個以上的幣種組成
num = dat.groupby('Date')['symbol'].count().reset_index().rename(columns={'symbol':'count'})
date = num[num['count']>=150]['Date'].values[0]
dat1 = dat[dat['Date']>= date]
dat1['Year'] = pd.to_datetime(dat1['Date']).dt.year
dat1['Month'] = pd.to_datetime(dat1['Date']).dt.month
dat1['Day'] = pd.to_datetime(dat1['Date']).dt.day

n = 30 #每一組取幾個
#小市值
#避免頻繁換手的問題，故每個月1號按照當天曝險值排序
sizeP1 = dat1.groupby(['Year','Month']).apply(lambda dat1: dat1[dat1['Day']==1].sort_values(by='size').iloc[:n])
sizeP1.index = np.arange(sizeP1.shape[0])
sizeP1 = dat1.merge(sizeP1[['symbol','Year','Month']], on=['symbol','Year','Month'], how='inner')
change_w1 = sizeP1.groupby('Date').apply(lambda sizeP1: sizeP1['cap_weighted']/sum(sizeP1['cap_weighted'])).reset_index()
sizeP1['cap_weighted'] = change_w1['cap_weighted'].values
sizeP1_ = sizeP1.groupby('Date').apply(lambda sizeP1: sum(sizeP1['cap_weighted']*sizeP1['return'])).reset_index().rename(columns={0:'Return'}) 
sizeP1_.describe()
len(sizeP1)/len(sizeP1_) #幣種數量/天 29.35

#其他風格因子投組方法以此類推