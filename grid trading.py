# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:48:55 2022

@author: KennyKuo
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

stop_loss_std=2.5
total_fee=0.0007
def capital_amount(N,Mean=0,Std=1):
    if N<2:
        return (round(norm.cdf(N,0,1)-norm.cdf(-N,Mean,Std),4))
    else:
        return 1

def refresh_position(position,price):
    temp=0
    for i in range(np.shape(position)[0]):
        v=position.loc[i]["Volume"]
        if v>0:
            c=position.loc[i]["Cost"]
            s_l=position.loc[i]["Stop loss"]
            l_i=position.loc[i]["Lock in gain"]
            if price<s_l:
                position.loc[i]["Profit"]=price*v*(1-total_fee)-c
                position.loc[i]["Volume"]=0
                temp=temp+position.loc[i]["Profit"]
            elif price>l_i:
                position.loc[i]["Profit"]=price*v*(1-total_fee)-c
                position.loc[i]["Volume"]=0
                temp=temp+position.loc[i]["Profit"]
        elif v<0:
            c=position.loc[i]["Cost"]
            s_l=position.loc[i]["Stop loss"]
            l_i=position.loc[i]["Lock in gain"]
            if price>s_l:
                position.loc[i]["Profit"]=price*v*(1+total_fee)-c
                position.loc[i]["Volume"]=0
                temp=temp+position.loc[i]["Profit"]
            elif price<l_i:
                position.loc[i]["Profit"]=price*v*(1+total_fee)-c
                position.loc[i]["Volume"]=0
                temp=temp+position.loc[i]["Profit"]
    return [position,temp]
        
def open_position(position,price,mean,vol,cash,internal):
    N_std=int(((price-mean)/vol)/internal)*internal
    if abs(N_std)>0:
        N=np.shape(position)[0]
        temp=pd.DataFrame(index=[N],columns=["Cost","Volume","Stop loss","Lock in gain","Profit"])
        new_volume=-cash*capital_amount(N_std)
        temp.loc[N]["Cost"]=new_volume+abs(new_volume*total_fee)
        temp.loc[N]["Volume"]=new_volume/price
        temp.loc[N]["Lock in gain"]=mean
        if new_volume>0:
            temp.loc[N]["Stop loss"]=mean-stop_loss_std*vol
        else:
            temp.loc[N]["Stop loss"]=mean+stop_loss_std*vol
        position=pd.concat([position,temp])
    return position
    
    
rawdata=pd.read_excel("C:/Users/KennyKuo/Downloads/BTCUSDT.xlsx")
four_hr_data=pd.DataFrame(index=range(int(np.shape(rawdata)[0]/4)),columns=["Open time","Open","High","Low","Close","Volume"])

i=0

for t in range(np.shape(rawdata)[0]):
    if t%4==3:
        four_hr_data.loc[i]["Open time"]=rawdata.loc[t-3]["Open_time"]
        four_hr_data.loc[i]["Open"]=rawdata.loc[t-3]["open"]
        four_hr_data.loc[i]["High"]=max(rawdata.loc[t-3:t]["high"])
        four_hr_data.loc[i]["Low"]=min(rawdata.loc[t-3:t]["low"])
        four_hr_data.loc[i]["Close"]=rawdata.loc[t]["close"]
        four_hr_data.loc[i]["Volume"]=min(rawdata.loc[t-3:t]["volume"])
        i=i+1
        
position=pd.DataFrame(columns=["Cost","Volume","Stop loss","Lock in gain","Profit"])
cash=100000
for t in range(1000):  #range(np.shape(four_hr_data)[0])
    if t>=42:
        price=four_hr_data.loc[t]["Close"]
        [position,net]=refresh_position(position,price)
        cash=cash+net
        vol=np.std(four_hr_data.loc[t-42:t]["Close"])
        mean=four_hr_data.loc[t-42]["Close"]
        position=open_position(position,price,mean,vol,cash,internal=0.5)
    