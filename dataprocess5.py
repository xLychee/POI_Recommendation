#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:43:57 2017

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

df['timed]