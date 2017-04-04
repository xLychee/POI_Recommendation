#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:33:53 2017

@author: xlychee
"""

import pandas as pd
import time
import random
import numpy as np
import datetime

df = pd.read_csv('../new_tuple.csv')


def timedif(time1,time2):
    time1 = datetime.datetime.strptime(time1,'%Y-%m-%dT%H:%M:%SZ')
    time2 = datetime.datetime.strptime(time2,'%Y-%m-%dT%H:%M:%SZ')
    return (time2-time1).total_seconds()

dtime = map(timedif,df['last_time'],df['now_time'])

df['timedelta']=dtime
   
user_total = df.groupby('user_id').size()
user_count = {}

train = []
test = []
for index,item in df.iterrows():
    if index%1000==0:
        print index
    user = item['user_id']
    allnum = user_total.loc[user]
    nownum = user_count.get(user,0)
    if nownum>2 and nownum>=0.8*allnum:
        test.append(item)
    else:
        train.append(item)
        user_count[user]=user_count.get(user,0)+1   

train = pd.DataFrame(train)
test = pd.DataFrame(test)


    
