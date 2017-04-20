#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:55:06 2017

@author: xlychee
"""


import pandas as pd
import time
import random
import numpy as np
import datetime
import math
from sklearn.model_selection import train_test_split


df = pd.read_csv('../train_new_41')
dftrain,dftest = train_test_split(df, test_size = 0.2)
dftrain,dfval =  train_test_split(dftrain, test_size = 0.25)
dftrain = df
class stellar(object):
    def __init__(self, wholedf):
        self.dim_feature = 40
        self.U = {}
        for index in wholedf['user_id'].unique():
            self.U[index] = np.random.rand(self.dim_feature)
        self.T = {}
        for index in wholedf['now_time'].unique():
            self.T[index] = np.random.rand(self.dim_feature)
        self.L1 = {}
        self.L2 = {}
        self.L3 = {}
        for index in set(wholedf['now_place'].unique()) | set(wholedf['last_place']):
            self.L1[index] = np.random.rand(self.dim_feature)
            self.L2[index] = np.random.rand(self.dim_feature)
            self.L3[index] = np.random.rand(self.dim_feature)
        
    
    def sigmoid(self,z):
        return 1.0 / (1.0+ math.exp(-z))
        
    def score(self,u,t,lq_2,lc_1,lc_2,lc_3,w):
        return lc_1.dot(u)+w*lc_2.dot(lq_2)+lc_3.dot(t)
    
        
            
    def train(self,df, reg=0.0001, iterations = 50, k = 40, learning_rate = 0.00001):
        
        num_tuples = df.shape[0]
        places = df['now_place'].unique()
        
        for ite in xrange(iterations):
            loss = []
            for i in xrange(num_tuples):
                
                u_id = df.iloc[i]['user_id']
                t_id = df.iloc[i]['now_time']
                lq_id = df.iloc[i]['last_place']
                lp_id = df.iloc[i]['now_place']
                w = df.iloc[i]['timedelta']
                w = float(w)/3600
                if w>=4:
                    w = 0.5 + 2/w
                else:
                    w = 1
                lnids = np.random.choice(places,k,replace = False)
                loss_t=0
                realk = k
                for ln_id in lnids:
                    if ln_id == lp_id:
                        realk -=1
                        continue
                    
                    u = self.U[u_id]
                    t = self.T[t_id]
                    lp1 = self.L1[lp_id]
                    lp2 = self.L2[lp_id]
                    lp3 = self.L3[lp_id]
                    
                    lq2 = self.L2[lq_id]
                    
                    ln1 = self.L1[ln_id]
                    ln2 = self.L2[ln_id]
                    ln3 = self.L3[ln_id]
                    
                    
                    fp = self.score(u,t,lq2,lp1,lp2,lp3,w)
                    fn = self.score(u,t,lq2,ln1,ln2,ln3,w)
                    delta = 1 - self.sigmoid(fp-fn)
                    
                    du = - delta*(lp1 - ln1) + reg * u
                    dt = - delta *(lp3 - ln3) + reg*t
                    dlq2 = -delta * w * (lp2 - ln2) + reg*lq2
                    dlp1 = -delta * u + reg*lp1
                    dlp2 = -delta * w * lq2 + reg*lp2
                    dlp3 = -delta * t + reg*lp3
                    dln1 = delta * u + reg*ln1
                    dln2 = delta * w * lq2 + reg*ln2
                    dln3 = delta * t + reg * ln3
                    
                    self.U[u_id] = np.maximum(0,u-learning_rate*du )
                    self.T[t_id] = np.maximum(0,t-learning_rate*dt )
                    self.L1[lp_id] = np.maximum(0,lp1 - learning_rate*dlp1 )
                    self.L2[lp_id] = np.maximum(0,lp2 - learning_rate*dlp2 )
                    self.L3[lp_id] = np.maximum(0,lp3 - learning_rate*dlp3 )
                    self.L1[ln_id] = np.maximum(0,ln1 - learning_rate*dln1 )
                    self.L2[ln_id] = np.maximum(0,ln2 - learning_rate*dln2 )
                    self.L3[ln_id] = np.maximum(0,ln3 - learning_rate*dln3 )
                    self.L2[lq_id] = np.maximum(0,lq2 - learning_rate*dlq2 )
                    
                    u = self.U[u_id]
                    t = self.T[t_id]
                    lp1 = self.L1[lp_id]
                    lp2 = self.L2[lp_id]
                    lp3 = self.L3[lp_id]
                    
                    lq2 = self.L2[lq_id]
                    
                    ln1 = self.L1[ln_id]
                    ln2 = self.L2[ln_id]
                    ln3 = self.L3[ln_id]
                    
                    fp = self.score(u,t,lq2,lp1,lp2,lp3,w)
                    fn = self.score(u,t,lq2,ln1,ln2,ln3,w)
                    
                    loss_t += - math.log(self.sigmoid(fp-fn))
    
                loss_t = loss_t / realk
                loss.append(loss_t)
                if i%1000 ==0 and i!=0:
                    print "ite: %d/%d, tuple:%d/%d, loss: %s" %(ite,iterations,i,num_tuples,loss[-1])
        return loss
    
    
    def test(self,df):
        num_tuples = df.shape[0]
        places = df['now_place'].unique()
        correct_num=0
        for i in xrange(num_tuples):   
            print i
            u_id = df.iloc[i]['user_id']
            t_id = df.iloc[i]['now_time']
            lq_id = df.iloc[i]['last_place']
            lp_id = df.iloc[i]['now_place']
            w = df.iloc[i]['timedelta']
            w = float(w)/3600
            if w>=4:
                w = 0.5 + 2/w
            else:
                w = 1
            u = self.U[u_id]
            t = self.T[t_id]
            
            lq2 = self.L2[lq_id]
            
            scores = []
            for lc_id in places:
                lc1 = self.L1[lc_id]
                lc2 = self.L2[lc_id]
                lc3 = self.L3[lc_id]
                s = self.score(u,t,lq2,lc1,lc2,lc3,w)
                scores.append((lc_id,s))
            scores.sort(key=lambda pair: pair[1], reverse = True)
            rec_plc = [pair[0] for pair in scores[:5]]
            if lp_id in rec_plc:
                correct_num+=1
                print str(i)+'/'+str(num_tuples)+"now precision:"+str(float(correct_num)/i)
        precision = float(correct_num)/num_tuples
        print "precision: ", precision
        return precision
        
model = stellar(df)
learning_rate = [0.00001, 0.00003, 0.0001, 0.0003, 0.001,\
                 0.003, 0.01, 0.03]

learning_rate = [0.001]
for lr in learning_rate:
    loss = model.train(df,learning_rate=lr)
    np.save('../L1',model.L1)
    np.save('../L2',model.L2)
    np.save('../L3',model.L3)
    np.save('../U',model.U)
    np.save('../T',model.T)
    #pval = model.test(dfval)
    #outfile = 'model' + str(learning_rate)
    
    
    #of.write(outfile+'\nloss:'+str(loss)+'\nprecision in train'+str(ptrain)\
    #         +'\nprecision in validation' + str(pval))
    
    ptrain = model.test(dftest)

#model.test(dftrain)
                
                
                
                
                
                
                
                