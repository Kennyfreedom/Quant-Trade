# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:22:24 2022

@author: bbcat
"""
import pandas as pd
from binance.client import Client

api="5luVkBdgG2urz3vA63oRksPz0ZOLvlxJ9NvDtQaA"
s_api="G1sAnU6C2XHDHzZPpPkgnKKRik34xQQTCgja25C2"
client=Client(api,s_api)
data=client.get_historical_klines("BTCUSDT", "1h","2017/7/1")

rawdata=pd.DataFrame(data,columns=["Open time","Open","High","Low","Close","Volume","Close time"\
                                   ,"Quote asset volume","Number of trade","Taker buy base asset volume"\
                                       ,"Taker buy quote asset volume","Ignore"])
