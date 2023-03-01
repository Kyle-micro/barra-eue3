# 重跑Cm要設Sample, o

import pandas as pd
import numpy as np
import statsmodels.api as sm

#全部ESTU
#讀因子loading處理好的資料
data = pd.read_csv('dacs_factor_done.csv').drop(columns=['Unnamed: 0'])
#排除 COCOS 當時沒有用到的 loading
ii = data[(data['symbol']=='COCOS') & (data['Date']=='2021-01-19')].index[0]
data = data[data.index!=ii]
ret = pd.read_csv('dacs_estu.csv').drop('Unnamed: 0',axis=1)
data = data.merge(ret[['Date','name','return']], on=['Date','name'], how='inner')
data['yearmon'] =  pd.to_datetime(data['Date']).dt.strftime('%Y%m') 
#計算因子報酬時，Y 要比 X 領先一期
data['leading_return'] = data.groupby('name')['return'].shift(-1) 
#拿掉每個幣別的最後一天
data = data.dropna()

#風格因子投組--每個月底的曝險值決定下個月的幣別
#設定要進行檢驗的投組
noRT = pd.DataFrame(columns=['Portfolios','Cap','Equal'])
RT = pd.DataFrame(columns=['Portfolios','Cap','Equal'])
rad_noRT = pd.DataFrame(columns=['Portfolios','Cap','Equal'])
rad_RT = pd.DataFrame(columns=['Portfolios','Cap','Equal'])
for title in ['Small Size','Poor Liquidity','High Volatility','High Centrality','Strong Momentum']:
    Sample = pd.read_csv("因子投組\ "+title+'.csv').drop('Unnamed: 0',axis=1)
                         
    #投組標準差 -- Common Factor Risk
    sam_size=540
    #sam_size=1500
    fac_ret = pd.read_csv('因子報酬敘述統計/ '+title+'_factor_return_adj.csv').set_index('Unnamed: 0')
    n = fac_ret.shape[0]-sam_size 
    Cmlist=[]     
    for i in range(n):
        a1=np.load('Common Factor Risk_Long\Cm_'+title+str(i)+'.npy')
        Cmlist.append(a1)
    #長期
    Sample = Sample[Sample['Date'] >= Sample['Date'].unique()[sam_size]]
    #短期
    #data = data[data['Date']>='2016-02-04']
    Sample = Sample.sort_values(['Sector','name'])
    Sample.index = np.arange(len(Sample))

    #市值加權投組風險============================================================================
    sdev_plist=[]
    o = 0 
    for date in np.sort(np.unique(Sample['Date'])):
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
    for date in np.sort(np.unique(Sample['Date'])):
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
    #daily return of Factor Portfolio
    ##排除COCOS的誇張報酬
    Sample = Sample[Sample['return']!=data[data['symbol']=='COCOS']['return'].max()] 
    Sample1 = Sample.groupby('Date').apply(lambda Sample: sum(Sample['cap_weighted']*Sample['return'])).reset_index().rename(columns={0:'cap_dr'}) 
    Sample1['e_dr'] = Sample.groupby('Date')['return'].mean().values
    Sample1['yearmon'] = pd.to_datetime(Sample1['Date']).dt.strftime('%Y%m')
    #monthly return of factor portfolio
    mfp = Sample1.groupby('yearmon').apply(lambda Sample1: (Sample1['cap_dr']+1).prod()-1).reset_index().rename(columns={0:'Cap_mr'})
    mfp['Equal_mr'] = Sample1.groupby('yearmon').apply(lambda Sample1: (Sample1['e_dr']+1).prod()-1).values
    tem = Sample1.groupby('yearmon').apply(lambda Sample1:Sample1['Date'].count()).reset_index()
    tem = tem[tem[0]>25]
    mfp = mfp[mfp['yearmon'].isin(tem['yearmon'])]

#=================Bias test 用月初的投組風險看這個月的報酬======================================================================= 
#每天的 Monthly risk of portfolio
    drp = [MT,ET]
    #monthly return of portfolio
    mp = [mfp['Cap_mr'].values,mfp['Equal_mr'].values]
    fig = ['Cap','Equal']

    noRlist = [title]
    Rlist = [title]
    rad_noRlist = [title]
    rad_Rlist = [title]
    for d,m,f in zip(drp, mp, fig): #先跑cap再跑equal
    #月初的 Monthly risk of portfolio
        d['yearmon'] = pd.to_datetime(d['date']).dt.strftime('%Y%m')
        d['day'] = pd.to_datetime(d['date']).dt.day
        mrp = d.groupby('yearmon').apply(lambda d: d.iloc[0,:])
        mrp = mrp[mrp['day']==1]
        
        z = pd.DataFrame(m/mrp['sdev_p'].values)
        z_sd = z.rolling(12).std()[11:] 
        #short_ze_std = ze.rolling(12).std()[11:]   
        bias = [ i for i in z_sd.values if i < 1+np.sqrt(1/6) and i>1-np.sqrt(1/6)]
        ratio = len(bias)/ z_sd.shape[0]
        noRlist.append(ratio)
        
        #Robust
        zR = []
        for i in np.array(m)/np.array(mrp['sdev_p']):
            zR.append(max(-3,min(3,i)))
        zR = pd.DataFrame(zR)
        zR_sd = zR.rolling(12).std()[11:] 
        #short_zeR_std = zeR.rolling(12).std()[11:]  
        biasR = [ i for i in zR_sd.values if i < 1+np.sqrt(1/6) and i>1-np.sqrt(1/6)]
        ratioR = len(biasR)/zR_sd.shape[0]
        Rlist.append(ratioR)
        
    #-------------------------------------圖---------------------------------------
        #無Robust
        z_sd.insert(0,'date',mrp['yearmon'].values[11:])
        #有Robust
        zR_sd.insert(0,'date',mrp['yearmon'].values[11:])
    
        #無Robust
        import matplotlib.pyplot as plt
        plt.style.use('seaborn')
        plt.figure(figsize = (12,8)).subplots_adjust(top=0.8)
        x = np.arange(len(z_sd))
        plt.scatter( x, z_sd[0], color='red' ,label='Long Term')
        #plt.scatter( Cap['date'], Cap['0_y'], color='gold' ,label='Long Term' ,marker='^')
        plt.axhline(1+np.sqrt(1/6),color="blue",linestyle='--')
        plt.axhline(1-np.sqrt(1/6),color="blue",linestyle='--')
        plt.axhline(1,color="black",linestyle='--')
        xlist = list(z_sd['date'])[::int(len(z_sd['date'])/8-1)]
        xx = list(x[::int(len(z_sd['date'])/8-1)])
        plt.xticks(xx, xlist,fontsize=13, rotation=30)
        plt.yticks(fontsize=13)
        plt.ylim(round(min(z_sd[0])-0.1,1),round(max(z_sd[0])+0.1,1)) 
        plt.legend(loc=1)
        plt.xlabel('Date', fontsize=13)
        plt.ylabel('b value', fontsize=13)
        plt.title(f+' Weight Bias Test', fontsize=16)
        plt.savefig('Bias test/'+title+' '+f+' Weight Bias Test.png')
    
        #有Robust
        import matplotlib.pyplot as plt
        plt.style.use('seaborn')
        plt.figure(figsize = (12,8)).subplots_adjust(top=0.8)
        x = np.arange(len(zR_sd))
        plt.scatter( x, zR_sd[0], color='red' ,label='Long Term')
        #plt.scatter( Cap['date'], Cap['0_y'], color='gold' ,label='Long Term' ,marker='^')
        plt.axhline(1+np.sqrt(1/6),color="blue",linestyle='--')
        plt.axhline(1-np.sqrt(1/6),color="blue",linestyle='--')
        plt.axhline(1,color="black",linestyle='--')
        xlist = list(zR_sd['date'])[::int(len(zR_sd['date'])/8-1)]
        xx = list(x[::int(len(zR_sd['date'])/8-1)])
        plt.xticks(xx, xlist,fontsize=13, rotation=30)
        plt.yticks(fontsize=13)
        plt.ylim(round(min(z_sd[0])-0.1,1),round(max(z_sd[0])+0.1,1))
        plt.legend(loc=1)
        plt.xlabel('Date', fontsize=13)
        plt.ylabel('b value', fontsize=13)
        plt.title(f+' Weight Robust Bias Test', fontsize=16)
        plt.savefig('Bias test/'+title+' '+f+' Weight Robust Bias Test.png')
    
    #RAD------------------------------------------------------
        #無Robust
        rad_noRlist.append(abs(z_sd[0]-1).mean())
        #有Robust
        rad_Rlist.append(abs(zR_sd[0]-1).mean()) 
        
    RT.loc[len(RT)] = Rlist
    noRT.loc[len(noRT)] = noRlist
    rad_RT.loc[len(rad_RT)] = rad_Rlist
    rad_noRT.loc[len(rad_noRT)] = rad_noRlist
RT.to_csv('Bias test/Robust bias.csv')
noRT.to_csv('Bias test/noRobust bias.csv')
rad_RT.to_csv('Bias test/Robust RAD.csv')
rad_noRT.to_csv('Bias test/noRobust RAD.csv')


'''
data = data[data['return']!=data[data['symbol']=='COCOS']['return'].max()] #排除COCOS的誇張報酬
cap = data.groupby('Date').apply(lambda data: sum(data['return'] * data['cap_weighted'])).reset_index().rename(columns={0:'dr'})
equ = data.groupby('Date')['return'].mean().reset_index().rename(columns={'return':'dr'})
cap['yearmon'] = pd.to_datetime(cap['Date']).dt.strftime('%Y%m')
capm = cap.groupby('yearmon').apply(lambda cap: (cap['dr']+1).prod()-1).reset_index().rename(columns={0:'mr'})
equ['yearmon'] = pd.to_datetime(equ['Date']).dt.strftime('%Y%m')
equm = equ.groupby('yearmon').apply(lambda cap: (cap['dr']+1).prod()-1).reset_index().rename(columns={0:'mr'})
'''







