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
lock_in=0.03 #停利
total_fee=0.0007

workspace="C://Users//KennyKuo//Downloads//Grid_trading//"

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
def refresh_position(position1,position2,price,time,wait,equity,condition=0):

    con_idx=price.index
    net=0
    realize=0
    unrealize=0
    cost=0
    if (wait<-0.1)&(condition==0):
        for i in con_idx:
            p=float(price[i])
            if np.shape(position1)[0]>0:
                term_p=(position1["Constract"]==i)&(position1["Volume"]>0)
                if np.shape(term_p[term_p==True])[0]>0:
                    position1.loc[term_p,"C_Time"]=time.copy()
                    position1.loc[term_p,"Fee"]=position1.loc[term_p,"Fee"]+p*position1.loc[term_p,"Volume"]*total_fee
                    position1.loc[term_p,"Profit"]=p*position1.loc[term_p,"Volume"]-position1.loc[term_p,"Cost"]-position1.loc[term_p,"Fee"]
                    position1.loc[term_p,"C_price"]=p
                    position1.loc[term_p,"Volume"]=0
                    net=net+sum(position1.loc[term_p,"Profit"])
                    realize=realize+sum(position1.loc[term_p,"Profit"])+sum(position1.loc[term_p,"Cost"])
                    
                    position2=pd.concat([position2,position1.loc[term_p]])
                    position1=position1.drop(position1.loc[term_p].index)
                    
                term_n=(position1["Constract"]==i)&(position1["Volume"]<0)
                if np.shape(term_n[term_n==True])[0]>0:
                    position1.loc[term_n,"C_Time"]=time.copy()
                    position1.loc[term_n,"Fee"]=position1.loc[term_n,"Fee"]-p*position1.loc[term_n,"Volume"]*total_fee
                    position1.loc[term_n,"Profit"]=p*position1.loc[term_n,"Volume"]-position1.loc[term_n,"Cost"]-position1.loc[term_n,"Fee"]
                    position1.loc[term_n,"C_price"]=p
                    position1.loc[term_n,"Volume"]=0
                    net=net+sum(position1.loc[term_n,"Profit"])
                    realize=realize+sum(position1.loc[term_n,"Profit"])-sum(position1.loc[term_n,"Cost"])
                    
                    position2=pd.concat([position2,position1.loc[term_n]])
                    position1=position1.drop(position1.loc[term_n].index)
            
            term_p2=(position1["Constract"]==i)&(position1["Volume"]>0)
            term_n2=(position1["Constract"]==i)&(position1["Volume"]<0)
            if (np.shape(term_p2[term_p2==True])[0]>0)|(np.shape(term_n2[term_n2==True])[0]>0):
                print("ERROR")
                input()
            
        unrealize=0
        cost=0
        condition=24
        wait=wait*0.9
    else:
        for i in con_idx:
            p=float(price[i])
            if np.shape(position1)[0]>0:
                term_p=(position1["Constract"]==i)&(position1["Volume"]>0) & ((position1["Stop loss"]>p) | (position1["Lock in gain"]<p))
                if np.shape(term_p[term_p==True])[0]>0:
                    position1.loc[term_p,"C_Time"]=time.copy()
                    position1.loc[term_p,"Fee"]=position1.loc[term_p,"Fee"]+p*position1.loc[term_p,"Volume"]*total_fee
                    position1.loc[term_p,"Profit"]=p*position1.loc[term_p,"Volume"]-position1.loc[term_p,"Cost"]-position1.loc[term_p,"Fee"]
                    position1.loc[term_p,"C_price"]=p
                    position1.loc[term_p,"Volume"]=0
                    net=net+sum(position1.loc[term_p,"Profit"])
                    realize=realize+sum(position1.loc[term_p,"Profit"])+sum(position1.loc[term_p,"Cost"])
                    
                    position2=pd.concat([position2,position1.loc[term_p]])
                    position1=position1.drop(position1.loc[term_p].index)
                    
                term_n=(position1["Constract"]==i)&(position1["Volume"]<0) & ((position1["Stop loss"]<p) | (position1["Lock in gain"]>p))
                if np.shape(term_n[term_n==True])[0]>0:
                    position1.loc[term_n,"C_Time"]=time.copy()
                    position1.loc[term_n,"Fee"]=position1.loc[term_n,"Fee"]-p*position1.loc[term_n,"Volume"]*total_fee
                    position1.loc[term_n,"Profit"]=p*position1.loc[term_n,"Volume"]-position1.loc[term_n,"Cost"]-position1.loc[term_n,"Fee"]
                    position1.loc[term_n,"C_price"]=p
                    position1.loc[term_n,"Volume"]=0
                    net=net+sum(position1.loc[term_n,"Profit"])
                    realize=realize+sum(position1.loc[term_n,"Profit"])-sum(position1.loc[term_n,"Cost"])
                    
                    position2=pd.concat([position2,position1.loc[term_n]])
                    position1=position1.drop(position1.loc[term_n].index)
            
            term_p2=(position1["Constract"]==i)&(position1["Volume"]>0)
            term_n2=(position1["Constract"]==i)&(position1["Volume"]<0)
            
            unrealize=unrealize+p*sum(position1.loc[term_p2,"Volume"])*(1-total_fee)-sum(position1.loc[term_p2,"Fee"])-sum(position1.loc[term_p2,"Cost"])+\
                (-sum(position1.loc[term_n2,"Cost"])+p*sum(position1.loc[term_n2,"Volume"])*(1+total_fee)-sum(position1.loc[term_n2,"Fee"]))
            cost=cost+sum(position1.loc[term_p2,"Cost"])-sum(position1.loc[term_n2,"Cost"])
            condition=0

    return [position1,position2,net,realize,unrealize,cost,wait,condition]
        
# %%
def open_position(idx,position,time,price,mean,vol,cash,s=start,e=end,intvl=interval):
    N_std=int(((price-mean)/vol)/intvl)*intvl
    pos_cost=0
    if abs(N_std)>=s and abs(N_std)<=e and N_std<0 :#and N_std<0 僅作多單
        N=np.shape(position)[0]
        temp=pd.DataFrame(index=[N],columns=["Constract","O_Time","O_price","C_Time","C_price","Proportion","Cost","Volume","Stop loss","Lock in gain","Fee","Profit","Cash"])
        if price>mean:
            new_volume=-min(cash*capital_amount(N_std),max_position)
            
            temp.loc[N,"Stop loss"]=price*(1+stop_loss) #mean+stop_loss_std*vol
            temp.loc[N,"Lock in gain"]=price*(1-lock_in) #mean+lock_in_std*vol
            
        else:
            new_volume=min(cash*capital_amount(N_std),max_position)
            temp.loc[N,"Stop loss"]=price*(1-stop_loss)
            temp.loc[N,"Lock in gain"]=price*(1+lock_in)
        temp.loc[N,"Constract"]=idx
        temp.loc[N,"O_Time"]=time
        temp.loc[N,"O_price"]=price
        temp.loc[N,"Proportion"]=f"{N_std}_{np.round(capital_amount(N_std),4)}"
        temp.loc[N,"Cost"]=new_volume
        temp.loc[N,"Fee"]=abs(new_volume*total_fee)
        temp.loc[N,"Volume"]=new_volume/price
        pos_cost=pos_cost+abs(new_volume)
        temp.loc[N,"Cash"]=cash-abs(new_volume)-temp.loc[N,"Fee"]
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
    rawdata=pd.read_csv(workspace+"Price.csv")
    v_rawdata=pd.read_csv(workspace+"volume.csv")
   
    hour_data=rawdata.drop("Open time", 1).apply(lambda x:x.astype(float))
    volume_data=(hour_data*v_rawdata.drop("Open time", 1)).apply(lambda x:x.astype(float))
    
    momt_mean=np.mean(((hour_data[1:].reset_index(drop=True)-hour_data[0:-1])/hour_data[0:-1]))
    momt_std=np.std(((hour_data[1:].reset_index(drop=True)-hour_data[0:-1])/hour_data[0:-1]))
            
    position1=pd.DataFrame(columns=["Constract","O_Time","O_price","C_Time","C_price","Proportion","Cost","Volume","Stop loss","Lock in gain","Fee","Profit","Cash"])
    position2=pd.DataFrame(columns=["Constract","O_Time","O_price","C_Time","C_price","Proportion","Cost","Volume","Stop loss","Lock in gain","Fee","Profit","Cash"])
    bs=pd.DataFrame(index=range(np.shape(hour_data)[0]),columns=["Time","BTC","Realize","Unrealize","Cost","Gain/Loss","Cash","Total"])
    
    t1=0
    t2=0
    wait=0
    equity=0
    condition=0
    for t in range(2000):  #range(np.shape(hour_data)[0])
        if t>72:
            if t%100 == 0:
                print("%d /"%t,"42600")
                
            #本日市場資訊
            #per=((hour_data.loc[t]-hour_data.loc[t-72])/hour_data.loc[t-72]).dropna()
            
            mean=(np.mean(hour_data.loc[t-72:t])).dropna()
            vol=((np.max(hour_data.loc[t-72:t])-np.min(hour_data.loc[t-72:t]))/2).dropna()
            
            price=hour_data.loc[t].dropna()
            volume=volume_data.loc[t-1].dropna()
            min_idx1=(hour_data.loc[t-72].dropna().index).intersection(hour_data.loc[t-24].dropna().index)
            min_idx2=(volume_data.loc[t-73].dropna().index).intersection(volume_data.loc[t-25].dropna().index)
            min_idx=min_idx1.intersection(min_idx2)
            #開倉條件
            #open_con1=(abs(per-momt_mean[per.index])<1*momt_std[per.index])
            
            daypro=((hour_data.loc[t,min_idx]-hour_data.loc[t-72,min_idx])/hour_data.loc[t-72,min_idx]).dropna()
            daypro3=((hour_data.loc[t,min_idx]-hour_data.loc[t-24,min_idx])/hour_data.loc[t-24,min_idx]).dropna()
            
            open_con2=(abs(daypro3)<abs(daypro))&((daypro3*daypro)>0)
            open_con3=(cash>10)
            open_con4=(volume/60>max_position)
            
            open_con5=(np.mean(volume_data.loc[t-25:t-1,min_idx]).dropna()<np.mean(volume_data.loc[t-73:t-1,min_idx]).dropna())
            
            #更新本日損益
            st1=time.time()
            [position1,position2,net,realize,unrealize,cost,wait,condition]=refresh_position(position1,position2,price,rawdata.loc[t,"Open time"].copy(),wait,equity,condition)
            ed1=time.time()
            t1=t1+(ed1-st1)
            
            #print("%d"%net,"%d"%realize,"%d"%unrealize)
            bs.loc[t,"Time"]=rawdata.loc[t,"Open time"]
            bs.loc[t,"BTC"]=hour_data.loc[t,"BTCUSDT"]
            bs.loc[t,"Realize"]=realize
            bs.loc[t,"Unrealize"]=unrealize
            bs.loc[t,"Cost"]=cost
            bs.loc[t,"Gain/Loss"]=net
            bs.loc[t,"Cash"]=cash
            bs.loc[t,"Total"]=realize+unrealize+cost+cash
            equity=realize+unrealize+cost+cash
            if abs(cost)>0:
                wait=unrealize/abs(cost)
            else:
                wait=0
            
            cash=cash+realize
            
            #開倉
            if condition==0:
                open_con= open_con2 & open_con3 & open_con4 & open_con5
                idx=select_constract(open_con,volume)
                
                st2=time.time()
                for i in idx:
                    [position1,pos_cost]=open_position(i,position1,rawdata.loc[t]["Open time"],float(price[i]),float(mean[i]),float(vol[i]),cash)
                    cash=cash-pos_cost
                position1.reset_index(inplace=True, drop=True)
                ed2=time.time()
                t2=t2+(ed2-st2)
            else:
                condition=condition-1
            
    #bs["Time"]=rawdata["Open time"]
    bs=bs.dropna(axis=0)
    
    reidx=pd.Index(i.strftime('%Y/%m/%d %H')for i in pd.to_datetime(bs["Time"], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')) #%Y/%m/%d %H:%M:%S'
    bs=bs.set_index(reidx)
    fig=plt.figure(figsize=(15,12))
    ax1=fig.add_subplot(3, 1, 1)
    ax2=fig.add_subplot(3, 1, 2)
    ax3=fig.add_subplot(3, 1, 3)
    
    ax1=(bs["Total"]).plot(c='black',grid=True,ax=ax1)
    #ax1=ax1.plot(bs["Time"],bs["Total"])
    #ax2=ax2.plot(bs["Time"],bs["Total"]-bs["Total"].cummax())
    ax2=((bs["Total"]-bs["Total"].cummax())/bs["Total"].cummax()).plot(grid=True,ax=ax2)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=1))
    ax3=(bs["BTC"]).plot(c='black',grid=True,ax=ax3)
    #ax2.fill_between(bs.index,0,((bs["Total"]-bs["Total"].cummax())/bs["Total"].cummax()),facecolor='red')
    ed = time.time()
    print("refresh_執行時間：%f 秒"%t1)
    print("open_執行時間：%f 秒"%t2)
    print("執行時間：%f 秒"%(ed - st))