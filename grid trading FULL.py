# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:48:55 2022

@author: KennyKuo
"""

import time
import pandas as pd
import numpy as np
from dateutil import tz 
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import ticker

style.use('ggplot')
style.use('seaborn')
TW = tz.gettz('Asia/Taipei')
Utc=tz.gettz('utc')
constract="OMGUSDT"

start=0.25 #開倉門檻
end=1 #最大開倉門檻
interval=0.25 #網格區間
stop_loss=0.05 #停損
lock_in=0.1 #停利
total_fee=0.0007

cash=10000
max_position=cash/10 #單筆倉位最大資金

# %%
def percent_of_capital(N,Mean=0,Std=1):
    return abs(round(norm.cdf(N,0,1)-norm.cdf(-N,Mean,Std),4))

# %%    
def capital_amount(N,Mean=0,Std=1,s=start,e=end,intvl=interval):
    temp=0
    if abs(N)<s:
        return 0
    elif abs(N)>=s and abs(N)<=e:
        temp=0
        for i in range(int((e-s)/intvl)+1):
            temp=temp+percent_of_capital(s+i*intvl)
        return (percent_of_capital(N)/temp)
    else:
        return 0
    
# %%
def refresh_position(position,price,time):

    con_idx=price.index
    net=0
    realize=0
    unrealize=0
    for i in con_idx:
        p=float(price[i])
        term_p=(position["Constract"]==i)&(position["Volume"]>0) & ((position["Stop loss"]>p) | (position["Lock in gain"]<p))
        term_n=(position["Constract"]==i)&(position["Volume"]<0) & ((position["Stop loss"]<p) | (position["Lock in gain"]>p))
        
        position.loc[term_p,"C_Time"]=time
        position.loc[term_p,"Fee"]=position.loc[term_p,"Fee"]+p*position[term_p]["Volume"]*total_fee
        position.loc[term_p,"Profit"]=p*position[term_p]["Volume"]-position[term_p]["Cost"]-position.loc[term_p,"Fee"]
        position.loc[term_p,"C_price"]=p
        position.loc[term_p,"Volume"]=0
        position.loc[term_n,"C_Time"]=time
        position.loc[term_n,"Fee"]=position.loc[term_n,"Fee"]-p*position[term_n]["Volume"]*total_fee
        position.loc[term_n,"Profit"]=p*position[term_n]["Volume"]-position[term_n]["Cost"]-position.loc[term_n,"Fee"]
        position.loc[term_n,"C_price"]=p
        position.loc[term_n,"Volume"]=0
        
        term_p2=(position["Constract"]==i)&(position["Volume"]>0)
        term_n2=(position["Constract"]==i)&(position["Volume"]<0)
        
        net=net+sum(position.loc[term_p,"Profit"])+sum(position[term_n]["Profit"])
        realize=realize+sum(position.loc[term_p,"Profit"])+sum(position[term_n]["Profit"])+sum(position.loc[term_p,"Cost"])-sum(position[term_n]["Cost"])
        unrealize=unrealize+p*sum(position.loc[term_p2,"Volume"])*(1-total_fee)-sum(position.loc[term_p2,"Fee"])+\
            (-2*sum(position[term_n2]["Cost"])+p*sum(position.loc[term_n2,"Volume"])*(1+total_fee)-sum(position.loc[term_n2,"Fee"]))

    return [position,net,realize,unrealize]
        
# %%
def open_position(idx,position,time,price,mean,vol,cash,s=start,e=end,intvl=interval):
    N_std=int(((price-mean)/vol)/intvl)*intvl
    pos_cost=0
    if abs(N_std)>=s and abs(N_std)<=e : #and N_std<0 僅作多單
        N=np.shape(position)[0]
        temp=pd.DataFrame(index=[N],columns=["Constract","O_Time","O_price","C_Time","C_price","Proportion","Cost","Volume","Stop loss","Lock in gain","Fee","Profit","Cash"])
        if price>mean:
            new_volume=-min(cash*capital_amount(N_std),max_position)
            
            temp.loc[N]["Stop loss"]=price*(1+stop_loss) #mean+stop_loss_std*vol
            temp.loc[N]["Lock in gain"]=price*(1-lock_in) #mean+lock_in_std*vol
            
        else:
            new_volume=min(cash*capital_amount(N_std),max_position)
            temp.loc[N]["Stop loss"]=price*(1-stop_loss)
            temp.loc[N]["Lock in gain"]=price*(1+lock_in)
        temp.loc[N]["Constract"]=idx
        temp.loc[N]["O_Time"]=time
        temp.loc[N]["O_price"]=price
        temp.loc[N]["Proportion"]=f"{N_std}_{np.round(capital_amount(N_std),4)}"
        temp.loc[N]["Cost"]=new_volume
        temp.loc[N]["Fee"]=abs(new_volume*total_fee)
        temp.loc[N]["Volume"]=new_volume/price
        pos_cost=pos_cost+abs(new_volume)
        temp.loc[N]["Cash"]=cash-abs(new_volume)-temp.loc[N]["Fee"]
        position=pd.concat([position,temp])
        
    
    return [position,pos_cost]

# %%
def select_constract(open_con,volume):
    
    if np.shape(open_con)[0]<=5:
        selection=open_con[open_con==True].index
    
    else:
        ori=open_con[open_con==True].index
        selection=volume[ori].sort_values(ascending=False)[0:4].index
        
    return selection    
    
# %%    
if __name__=='__main__':
    
    st = time.time()
    rawdata=pd.read_csv("C://Users//KennyKuo//Downloads//Grid_trading//price.csv")
    v_rawdata=pd.read_csv("C://Users//KennyKuo//Downloads//Grid_trading//volume.csv")
   
    hour_data=rawdata.drop("Open time", 1).apply(lambda x:x.astype(float))
    volume_data=(hour_data*v_rawdata.drop("Open time", 1)).apply(lambda x:x.astype(float))
    
    momt_mean=np.mean(((hour_data[1:].reset_index(drop=True)-hour_data[0:-1])/hour_data[0:-1]))
    momt_std=np.std(((hour_data[1:].reset_index(drop=True)-hour_data[0:-1])/hour_data[0:-1]))
            
    position=pd.DataFrame(columns=["Constract","O_Time","O_price","C_Time","C_price","Proportion","Cost","Volume","Stop loss","Lock in gain","Fee","Profit","Cash"])
    bs=pd.DataFrame(index=range(np.shape(hour_data)[0]),columns=["Time","Realize","Unrealize","Gain/Loss","Cash","Total"])
    
    
    for t in range(38145,42600):  #range(np.shape(hour_data)[0])
        if t>72:
            print("%d /"%t,"42600")
            per=((hour_data.loc[t]-hour_data.loc[t-72])/hour_data.loc[t-72]).dropna()
            vol=(np.std(hour_data.loc[t-72:t])).dropna()
            mean=hour_data.loc[t-72]
            
            price=hour_data.loc[t].dropna()
            volume=volume_data.loc[t-1].dropna()
            min_idx=(hour_data.loc[t-72].dropna().index).intersection(hour_data.loc[t-24].dropna().index)
            #open_con1=(abs(per-momt_mean[per.index])<1*momt_std[per.index])
            open_con2=abs(((hour_data.loc[t,min_idx]-hour_data.loc[t-24,min_idx])/hour_data.loc[t-24,min_idx]).dropna())\
                        <abs(((hour_data.loc[t,min_idx]-hour_data.loc[t-72,min_idx])/hour_data.loc[t-72,min_idx]).dropna())
            open_con3=(cash>10)
            open_con4=(volume/60>max_position)
            open_con5=(np.mean(volume_data.loc[t-25:t-1,min_idx]).dropna()<np.mean(volume_data.loc[t-73:t-1,min_idx]).dropna())
            
            
           
            [position,net,realize,unrealize]=refresh_position(position,price,rawdata.loc[t]["Open time"])
            #print("%d"%net,"%d"%realize,"%d"%unrealize)
            bs.loc[t]["Realize"]=realize
            bs.loc[t]["Unrealize"]=unrealize
            bs.loc[t]["Gain/Loss"]=net
            bs.loc[t]["Cash"]=cash
            bs.loc[t]["Total"]=realize+unrealize+cash
            
            cash=cash+realize
            
            open_con= open_con2 & open_con3 & open_con4 & open_con5
            idx=select_constract(open_con,volume)
            for i in idx:
                [position,pos_cost]=open_position(i,position,rawdata.loc[t]["Open time"],float(price[i]),float(mean[i]),float(vol[i]),cash)
                cash=cash-pos_cost
    bs["Time"]=rawdata["Open time"]
    bs=bs.dropna(axis=0)
    
    reidx=pd.Index(i.strftime('%Y/%m/%d %H:%M:%S')for i in pd.to_datetime(bs["Time"], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei'))
    bs=bs.set_index(reidx)
    
    fig=plt.figure(figsize=(15,8))
    ax1=fig.add_subplot(2, 1, 1)
    ax2=fig.add_subplot(2, 1, 2)
    
    ax1=(bs["Total"]).plot(c='black',grid=True,ax=ax1)
    #ax1=ax1.plot(bs["Time"],bs["Total"])
    #ax2=ax2.plot(bs["Time"],bs["Total"]-bs["Total"].cummax())
    ax2=((bs["Total"]-bs["Total"].cummax())/bs["Total"].cummax()).plot(grid=True,ax=ax2)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=1))
    #ax2.fill_between(bs.index,0,((bs["Total"]-bs["Total"].cummax())/bs["Total"].cummax()),facecolor='red')
    
    ed = time.time()
    print("執行時間：%f 秒"%(ed - st))