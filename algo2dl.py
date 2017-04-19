#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:06:13 2017

@author: xlychee
"""


import pandas as pd
import time
import random
import numpy as np
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import pickle
from sklearn.model_selection import train_test_split
from datetime import timedelta

def curtime():
    return time.asctime(time.localtime(time.time()))

print curtime()+" Program begin "


df = pd.read_csv('../train_new')

dftrain,dftest = train_test_split(df, test_size = 0.2)

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
        
    

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#dim_features = 100

w_p = weight_variable([num_places,100])
b_p = bias_variable([100])
#x_p = tf.placeholder(tf.float32, shape=[None, num_places])
x_p = tf.placeholder(tf.int32)
x_p_1 = tf.one_hot(x_p,depth = num_places)
x_p_2 = tf.reshape(x_p_1, [-1, num_places])
feature_p = tf.matmul(x_p_2, w_p) +b_p

w_u = weight_variable([num_users, 100])
b_u = bias_variable([100])
x_u = tf.placeholder(tf.float32, shape=[None, num_users])
feature_u = tf.matmul(x_u, w_u) +b_u

w_t = weight_variable([num_times, 30])
b_t = bias_variable([30])
x_t = tf.placeholder(tf.float32, shape=[None, num_times])
feature_t = tf.matmul(x_t, w_t) +b_t


w1 = weight_variable([100,200])
b1 = bias_variable([200])
layer1 = tf.matmul(feature_p,w1)+b1

w2 = weight_variable([100,200])
b2 = bias_variable([200])
layer1 += tf.matmul(feature_u,w2)+b2

w3 = weight_variable([30,200])
b3 = bias_variable([200])
layer1 += tf.matmul(feature_t,w3)+b3

layer1 = tf.nn.relu(layer1)

ww = weight_variable([200,num_places])
bb = bias_variable([num_places])

output = tf.matmul(layer1,ww) + bb 

y_ = tf.placeholder(tf.float32, shape=[None, num_places])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))

regularizers = tf.nn.l2_loss(w_p)+ tf.nn.l2_loss(w_u)\
    +  tf.nn.l2_loss(w_t) +  tf.nn.l2_loss(w1) \
    + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) \
    + tf.nn.l2_loss(ww)

loss = cross_entropy + regularizers
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())


print curtime()+" Training begin "
begintime = time.time() 
total_steps = 1000
for i in range(total_steps):
    u_id = map_users[df.iloc[i]['user_id']]
    lq_id = map_places[df.iloc[i]['last_place']]
    lp_id = map_places[df.iloc[i]['now_place']]
    t_id = map_times[df.iloc[i]['now_time']]
    x_u_batch = tf.one_hot([u_id],depth = num_users).eval()
    x_p_batch = tf.one_hot([lq_id],depth = num_places).eval()
    x_t_batch =  tf.one_hot([t_id],depth = num_times).eval()
    y_batch = tf.one_hot([lp_id],depth = num_places).eval()
    
    if i%10 == 0:
        used_time = str(timedelta(seconds=int(time.time()-begintime)))
        print "here", u_id, lq_id, lp_id, t_id
        train_accuracy=0
        test_accuracy = 0
        #train_accuracy = accuracy.eval(feed_dict={
        #        x:X_batch, y_: y_batch, keep_prob: 1.0})
        #test_accuracy = accuracy.eval(feed_dict={
        #        x:X_test, y_: y_test, keep_prob: 1.0})
        print("step %d/%d, used time %s, training accuracy %g, test accuracy %s" %(i,total_steps, used_time, train_accuracy, test_accuracy))
        #print "regularization:",reg_para*regularizers.eval(),"loss:",loss.eval(feed_dict={
        #        x:X_batch, y_: y_batch, keep_prob: 0.5})
    train_step.run(feed_dict={x_p: lq_id, x_u: x_u_batch, x_t: x_t_batch,y_: y_batch})

#prediction = (tf.argmax(output,1)).eval(feed_dict={x:test_X, keep_prob:1.0})


#m = 3
#t=tf.SparseTensor(indices=[1], values=[1], dense_shape=[len(places)])



















