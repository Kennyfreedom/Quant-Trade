# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:48:55 2022

@author: KennyKuo
"""

import time
import pandas as pd
import numpy as np
from scipy.stats import norm

start=0.25 #開倉門檻
end=1 #最大開倉門檻
interval=0.25 #網格區間
stop_loss_std=1.15 #停損
lock_in_std=0.1 #停利
total_fee=0.0007

cash=100000
max_position=cash/4 #單筆倉位最大資金

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

    term_p=(position["Volume"]>0) & ((position["Stop loss"]>price) | (position["Lock in gain"]<price))
    term_n=(position["Volume"]<0) & ((position["Stop loss"]<price) | (position["Lock in gain"]>price))
    
    position.loc[term_p,"C_Time"]=time
    position.loc[term_p,"Fee"]=position.loc[term_p,"Fee"]+price*position[term_p]["Volume"]*total_fee
    position.loc[term_p,"Profit"]=price*position[term_p]["Volume"]-position[term_p]["Cost"]-position.loc[term_p,"Fee"]
    position.loc[term_p,"C_price"]=price
    position.loc[term_p,"Volume"]=0
    position.loc[term_n,"C_Time"]=time
    position.loc[term_n,"Fee"]=position.loc[term_n,"Fee"]-price*position[term_n]["Volume"]*total_fee
    position.loc[term_n,"Profit"]=price*position[term_n]["Volume"]-position[term_n]["Cost"]-position.loc[term_n,"Fee"]
    position.loc[term_n,"C_price"]=price
    position.loc[term_n,"Volume"]=0
    
    term_p2=(position["Volume"]>0)
    term_n2=(position["Volume"]<0)
    
    net=sum(position.loc[term_p,"Profit"])+sum(position[term_n]["Profit"])
    realize=sum(position.loc[term_p,"Profit"])+sum(position[term_n]["Profit"])+sum(position.loc[term_p,"Cost"])-sum(position[term_n]["Cost"])
    unrealize=price*sum(position.loc[term_p2,"Volume"])*(1-total_fee)-sum(position.loc[term_p2,"Fee"])+\
        (-2*sum(position[term_n2]["Cost"])+price*sum(position.loc[term_n2,"Volume"])*(1+total_fee)-sum(position.loc[term_n2,"Fee"]))

    return [position,net,realize,unrealize]
        
# %%
def open_position(position,time,price,mean,vol,cash,s=start,e=end,intvl=interval):
    N_std=int(((price-mean)/vol)/intvl)*intvl
    pos_cost=0
    if abs(N_std)>=s and abs(N_std)<=e : #and N_std<0 僅作多單
        N=np.shape(position)[0]
        temp=pd.DataFrame(index=[N],columns=["O_Time","O_price","C_Time","C_price","Proportion","Cost","Volume","Stop loss","Lock in gain","Fee","Profit","Cash"])
        if price>mean:
            new_volume=-min(cash*capital_amount(N_std),max_position)
            temp.loc[N]["Stop loss"]=price*1.03 #mean+stop_loss_std*vol
            temp.loc[N]["Lock in gain"]=price*0.95 #mean+lock_in_std*vol
            
        else:
            new_volume=min(cash*capital_amount(N_std),max_position)
            temp.loc[N]["Stop loss"]=price*0.97
            temp.loc[N]["Lock in gain"]=price*1.05
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
if __name__=='__main__':
    
    st = time.time()
    rawdata=pd.read_excel("C:/Users/KennyKuo/Downloads/BTCUSDT.xlsx")
    four_hr_data=pd.DataFrame(index=range(int(np.shape(rawdata)[0]/4)),columns=["Open time","Open","High","Low","Close","close chg%","Volume"])
    i=0
    for t in range(np.shape(rawdata)[0]):
        if t%4==3:
            four_hr_data.loc[i]["Open time"]=rawdata.loc[t-3]["Open_time"]
            four_hr_data.loc[i]["Open"]=rawdata.loc[t-3]["open"]
            four_hr_data.loc[i]["High"]=max(rawdata.loc[t-3:t]["high"])
            four_hr_data.loc[i]["Low"]=min(rawdata.loc[t-3:t]["low"])
            four_hr_data.loc[i]["Close"]=rawdata.loc[t]["close"]
            four_hr_data.loc[i]["Volume"]=min(rawdata.loc[t-3:t]["volume"])
            four_hr_data.loc[i]["close chg%"]=(four_hr_data.loc[i]["Close"]-four_hr_data.loc[i]["Open"])/four_hr_data.loc[i]["Open"]
            i=i+1
    
    momt_mean=np.mean(four_hr_data["close chg%"])
    momt_std=np.std(four_hr_data["close chg%"])
            
    position=pd.DataFrame(columns=["O_Time","O_price","C_Time","C_price","Proportion","Cost","Volume","Stop loss","Lock in gain","Fee","Profit","Cash"])
    bs=pd.DataFrame(index=range(np.shape(four_hr_data)[0]),columns=["Time","Price","Mean","Vol","Condition","Realize","Unrealize","Gain/Loss","Cash","Total"])
    bs["Time"]=four_hr_data["Open time"]
    bs["Price"]=four_hr_data["Close"]
    for t in range(np.shape(four_hr_data)[0]):  #range(np.shape(four_hr_data)[0])
        if t>=42:
            per=four_hr_data.loc[t]["close chg%"]
            vol=np.std(four_hr_data.loc[t-42:t]["Close"])
            mean=four_hr_data.loc[t-42]["Close"]
            
            open_con1=(abs(per-momt_mean)<1*momt_std)
            open_con2=abs((four_hr_data.loc[t]["Close"]-four_hr_data.loc[t-18]["Close"])/four_hr_data.loc[t-18]["Close"])<abs((four_hr_data.loc[t]["Close"]-four_hr_data.loc[t-42]["Close"])/four_hr_data.loc[t-42]["Close"])
            
            price=four_hr_data.loc[t]["Close"]
            [position,net,realize,unrealize]=refresh_position(position,price,four_hr_data.loc[t]["Open time"])
            
            bs.loc[t]["Mean"]=mean
            bs.loc[t]["Vol"]=vol
            bs.loc[t]["Realize"]=realize
            bs.loc[t]["Unrealize"]=unrealize
            bs.loc[t]["Gain/Loss"]=net
            bs.loc[t]["Cash"]=cash
            bs.loc[t]["Total"]=realize+unrealize+cash
            
            cash=cash+realize
            
            open_con3=(cash>10)
            bs.loc[t]["Condition"]="%s %s %s"%(open_con1,open_con2,open_con3)
            
            if open_con1 and open_con2 and open_con3:
                [position,pos_cost]=open_position(position,four_hr_data.loc[t]["Open time"],price,mean,vol,cash)
                cash=cash-pos_cost
    bs=bs.dropna(axis=0)
    ed = time.time()
    print("執行時間：%f 秒"%(ed - st))
