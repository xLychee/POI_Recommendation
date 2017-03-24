#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:26:14 2017

@author: xlychee
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:04:12 2017

@author: xlychee
"""

import pandas as pd
import time
import random
import numpy as np
import datetime
from scipy import sparse


def curtime():
    return time.asctime(time.localtime(time.time()))

print curtime()+" Program begin"

df = pd.read_csv('../gowalla.txt', names=['user_id','time','lat','lon','place_id'],delim_whitespace=True);

userids =df['user_id']
placeids =df['place_id']
timeinfos = df['time']


gps = {}
for index,item in df.iterrows():
    pid = item['place_id']
    lat = item['lat']
    lon = item['lon']
    if pid in gps.keys():
        if gps[pid][0] != lat or gps[pid][1]!=lon:
            print 'wrong'
    else:
        gps[pid]=(lat,lon)



def timeid(timeinfo):
    a = datetime.datetime.strptime(timeinfo,'%Y-%m-%d %H:%M:%S');
    month = a.month-1
    weekday = 1 if a.isoweekday() == 6 or a.isoweekday() ==7 else 0
    hour=0
    if a.hour>=3 and a.hour<6:
        hour=0
    elif a.hour>=6 and a.hour<11:
        hour=1
    elif a.hour>=11 and a.hour<15:
        hour=2
    else:
        hour=3
    #print month,weekday,hour
    return month*8+weekday*4+hour

#timeids = np.array(map(timeid,timeinfos))

        
        
    
print curtime()+ "Program end"