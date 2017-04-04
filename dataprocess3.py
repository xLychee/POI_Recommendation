#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:47:10 2017

@author: xlychee
"""

import pandas as pd
import time
import random
import numpy as np
import datetime
from scipy import sparse


df = pd.read_csv('../deleted_data')


newdf = df.groupby('place_id').mean()

newdf.to_csv('../placedf.csv')