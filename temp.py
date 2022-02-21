import time
import pandas as pd
import numpy as np
import datetime as dt
import calendar
import ftx

client = ftx.FtxClient()

now=dt.datetime(2021,1,1,23,0,0)
start=int(dt.datetime(2021,9,24,23,0,0).timestamp())
end=int(dt.datetime(2021,12,31,23,0,0).timestamp())
interval=3600#seconds
Currency_type=pd.Series(["BTC","ETH","BNB","SOL","LTC","XRP","DOGE","SUSHI","LINK","BCH","AAVE", "TRX", "CHZ","UNI", "BAL", "SXP", "COMP","YFI","XAUT", "OMG","EDEN","OKB","1INCH", "WAVES","GRT","REEF","CEL"])


def future_mat_time(cur,time_con):
    cal_time=calendar.monthcalendar(time_con.year,time_con.month)
    last_Fri_time=cal_time[np.shape(cal_time)[0]-1][4] if cal_time[np.shape(cal_time)[0]-1][4] > 0 else cal_time[np.shape(cal_time)[0]-2][4]
    if (time_con.month<3)|((time_con.month==3)&(time_con.day<last_Fri_time)):
        cal=calendar.monthcalendar(time_con.year,3)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year,3,last_Fri)
    elif ((time_con.month>3)&(time_con.month<6))|((time_con.month==3)&(time_con.day>=last_Fri_time))|((time_con.month==6)&(time_con.day<last_Fri_time)):
        cal=calendar.monthcalendar(time_con.year,6)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year,6,last_Fri)
    elif ((time_con.month>6)&(time_con.month<9))|((time_con.month==6)&(time_con.day>=last_Fri_time))|((time_con.month==9)&(time_con.day<last_Fri_time)):
        cal=calendar.monthcalendar(time_con.year,9)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year,9,last_Fri)
    elif ((time_con.month>9)&(time_con.month<12))|((time_con.month==9)&(time_con.day>=last_Fri_time))|((time_con.month==12)&(time_con.day<last_Fri_time)):
        cal=calendar.monthcalendar(time_con.year,12)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year,12,last_Fri)
    else:
        cal=calendar.monthcalendar(time_con.year+1,1)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year+1,1,last_Fri)
    
    return f'{cur}-{temp:%Y%m%d}'
        

def daily_table(time_d):
    table=pd.DataFrame(columns=['Currency','spot_open', 'future_open','spot_close','future_close','Spread','abs_Spread'],index=Currency_type)
    for i in range(0,np.shape(Currency_type)[0]):
        try:
            mk_f=client.get_historical_data(market_name =future_mat_time(Currency_type[i],time_d),resolution =3600,start_time =time_d.timestamp(),end_time =(time_d).timestamp(),limit = 5000)
            df = pd.DataFrame(mk_f)
            df.index=[f'{future_mat_time(Currency_type[i],time_d)}']
            
            mk_s=client.get_historical_data(market_name =f'{Currency_type[i]}/USDT',resolution =3600,start_time =time_d.timestamp(),end_time =(time_d).timestamp(),limit = 5000)
            df2 = pd.DataFrame(mk_s)
            df2.index=[f'{Currency_type[i]}/USDT']
            
            table.loc[f'{Currency_type[i]}'][1]=df2.loc[f'{Currency_type[i]}/USDT']['open']
            table.loc[f'{Currency_type[i]}'][2]=df2.loc[f'{Currency_type[i]}/USDT']['close']
            table.loc[f'{Currency_type[i]}'][3]=df.loc[f'{future_mat_time(Currency_type[i],time_d)}']['open']
            table.loc[f'{Currency_type[i]}'][4]=df.loc[f'{future_mat_time(Currency_type[i],time_d)}']['close']
            table.loc[f'{Currency_type[i]}'][5]=(table.loc[f'{Currency_type[i]}'][3]-table.loc[f'{Currency_type[i]}'][1])/table.loc[f'{Currency_type[i]}'][1]
            table.loc[f'{Currency_type[i]}'][6]=abs(table.loc[f'{Currency_type[i]}'][5])
            table.loc[f'{Currency_type[i]}'][0]=f'{Currency_type[i]}'
        except:
            next
    table=table.dropna(axis=0,how='any')
    table=table.sort_values(by=['abs_Spread'],ascending=False)
    return table.sort_values(by=['abs_Spread'],ascending=False)

def balance_sheet(table,time_d):
    table_temp=table.head(5)
    Output_table=pd.DataFrame(columns=['Date','Currency', 'spot_cost','future_cost','spot_position','future_position','Total Spread','Maturity','Profit'])
    Output_table['spot_cost']=table_temp['spot_open']
    Output_table['future_cost']=table_temp['future_open']
    Output_table['spot_position']=100000/Output_table['spot_cost']
    Output_table['future_position']=100000/Output_table['future_cost']
    Output_table['Total Spread']=abs(Output_table['future_cost']-Output_table['spot_cost'])*Output_table[['spot_position','future_position']].max(axis=1)
    Output_table['Maturity']=future_mat(time_d)
    Output_table['Date']=time_d
    Output_table['Currency']=table_temp['Currency']
    return Output_table

def future_mat(time_con):
    cal_time=calendar.monthcalendar(time_con.year,time_con.month)
    last_Fri_time=cal_time[np.shape(cal_time)[0]-1][4] if cal_time[np.shape(cal_time)[0]-1][4] > 0 else cal_time[np.shape(cal_time)[0]-2][4]
    if (time_con.month<3)|((time_con.month==3)&(time_con.day<last_Fri_time)):
        cal=calendar.monthcalendar(time_con.year,3)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year,3,last_Fri)
    elif ((time_con.month>3)&(time_con.month<6))|((time_con.month==3)&(time_con.day>=last_Fri_time))|((time_con.month==6)&(time_con.day<last_Fri_time)):
        cal=calendar.monthcalendar(time_con.year,6)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year,6,last_Fri)
    elif ((time_con.month>6)&(time_con.month<9))|((time_con.month==6)&(time_con.day>=last_Fri_time))|((time_con.month==9)&(time_con.day<last_Fri_time)):
        cal=calendar.monthcalendar(time_con.year,9)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year,9,last_Fri)
    elif ((time_con.month>9)&(time_con.month<12))|((time_con.month==9)&(time_con.day>=last_Fri_time))|((time_con.month==12)&(time_con.day<last_Fri_time)):
        cal=calendar.monthcalendar(time_con.year,12)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year,12,last_Fri)
    else:
        cal=calendar.monthcalendar(time_con.year+1,1)
        last_Fri=cal[np.shape(cal)[0]-1][4] if cal[np.shape(cal)[0]-1][4] > 0 else cal[np.shape(cal)[0]-2][4]
        temp=dt.date(time_con.year+1,1,last_Fri)
    
    return temp+dt.timedelta(-1)

bs=balance_sheet(daily_table(dt.datetime.utcfromtimestamp(start)),dt.datetime.utcfromtimestamp(start))
profit=0
for i in range(1,int((end-start)/86400)):
    temp_d=start+i*86400
    dat=dt.datetime.utcfromtimestamp(temp_d)
    market_table=daily_table(dat)
    today_position=balance_sheet(market_table,dat)
    temp=[]
    temp1=[]
    for j in range(0,5):
        if bs.iloc[j][1] in today_position['Currency']:
            temp1.append(bs.iloc[j][1])
            next
        else:
            spot=market_table.loc[bs.iloc[j][1]][1]
            future=spot=market_table.loc[bs.iloc[j][1]][2]
            spot_cost=bs.loc[bs.iloc[j][1]][2]
            future_cost=bs.loc[bs.iloc[j][1]][3]
            if spot_cost<future_cost:
                amount=bs.loc[bs.iloc[j][1]][3]
                profit=profit+amount*((spot-spot_cost)-(future-future_cost))
            else:
                amount=bs.loc[bs.iloc[j][1]][4]
                profit=profit+amount*((future-future_cost)-(spot-spot_cost))
            temp.append(bs.iloc[j][1])
    today_position.loc[temp1]=bs.loc[temp1]
    bs=today_position