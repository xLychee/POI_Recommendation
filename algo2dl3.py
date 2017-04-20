#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:14:47 2017

@author: xlychee
"""

import tensorflow as tf
import tflearn
import numpy as np
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from sklearn import model_selection

import pandas as pd
import time
import random
import numpy as np
import datetime
import tensorflow as tf
#import matplotlib.pyplot as plt
import math
import pickle
from sklearn.model_selection import train_test_split
from datetime import timedelta



def curtime():
    return time.asctime(time.localtime(time.time()))

print curtime()+" Program begin "


df = pd.read_csv('../train_new_41')


df,dftest = train_test_split(df, test_size = 0.2)

sess = tf.InteractiveSession()

places = np.array(list(set(df['last_place'].unique()) |set(df['now_place'].unique())))
num_places = len(places)
map_places = {}
counter = 0
for p in places:
    if p not in map_places:
        map_places[p] = counter
        counter +=1


users = df['user_id'].unique()
num_users = len(users)
map_users = {}
counter = 0
for u in users:
    if u not in map_users:
        map_users[u]=counter
        counter+=1
        
times = df['now_time'].unique()
num_times = len(times)
map_times ={}
counter = 0
for t in times:
    if t not in map_times:
        map_times[t] = counter
        counter+=1



tf.reset_default_graph()

net_p = tflearn.input_data(shape = [None],dtype='int32')

net_p = tflearn.one_hot_encoding(net_p, n_classes=num_places)

net_p = tflearn.fully_connected(net_p,200,activation='linear',weights_init = 'normal', regularizer='L2')


net_u = tflearn.input_data(shape = [None],dtype='int32')

net_u = tflearn.one_hot_encoding(net_u, n_classes=num_users)

net_u = tflearn.fully_connected(net_u,200,activation='linear',weights_init = 'normal',regularizer='L2')


net_t = tflearn.input_data(shape = [None],dtype='int32')

net_t = tflearn.one_hot_encoding(net_t, n_classes=num_times)

net_t = tflearn.fully_connected(net_t,100,activation='linear',weights_init = 'normal',regularizer='L2')



net = tflearn.fully_connected(net_p,500,activation='Relu',weights_init = 'normal',regularizer='L2')\
+tflearn.fully_connected(net_u,500,activation='Relu',weights_init = 'normal',regularizer='L2')\
+tflearn.fully_connected(net_t,500,activation='Relu',weights_init = 'normal',regularizer='L2')

net = tflearn.activation(net,activation='Relu')

net = tflearn.fully_connected(net,500,activation='Relu',weights_init = 'normal',regularizer='L2')

net = tflearn.fully_connected(net,500,activation='Relu',weights_init = 'normal',regularizer='L2')

net = tflearn.fully_connected(net,500,activation='Relu',weights_init = 'normal',regularizer='L2')

net = tflearn.fully_connected(net,500,activation='Relu',weights_init = 'normal',regularizer='L2')

net = tflearn.fully_connected(net,500,activation='Relu',weights_init = 'normal',regularizer='L2')

net = tflearn.fully_connected(net, num_places,weights_init = 'normal',activation='softmax')


#net_y = tflearn.input_data(shape = [None],dtype='int32', name='1')


#net_yy = tflearn.one_hot_encoding(net_y, n_classes=num_places,name='2')

net = tflearn.regression(net, to_one_hot=True, n_classes = num_places, optimizer='adam',loss='categorical_crossentropy')

model = tflearn.DNN(net)

X_p = np.array(map(lambda x:map_places[x], df['last_place']))

X_u = np.array(map(lambda x:map_users[x], df['user_id']))

X_t = np.array(map(lambda x:map_times[x], df['now_time']))

y = np.array(map(lambda x:map_places[x], df['now_place']))

X_p_train = X_p[:-10000]
X_u_train = X_u[:-10000]
X_t_train = X_t[:-10000]
y_train = y[:-10000]

X_p_test = X_p[-10000:]
X_u_test = X_u[-10000:]
X_t_test = X_t[-10000:]
y_test = y[-10000:]

model.fit([X_p_train,X_u_train,X_t_train],y_train,validation_set=([X_p_test,X_u_test,X_t_test],y_test),
          n_epoch=20,batch_size=500,show_metric=True)

predictions = model.predict([X_p_test,X_u_test,X_t_test])

correct = 0
total_test = y_test.shape[0]
for i in xrange(y_test.shape[0]):
    pred = np.array(predictions[i]) 
    if y_test[i] in pred.argsort()[-5:]:
        correct+=1
print 'precision:', float(correct)/total_test







