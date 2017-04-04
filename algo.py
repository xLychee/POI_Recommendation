
import pandas as pd
import time
import random
import numpy as np
import datetime
import math

df = pd.read_csv('../train_new')

class stellar(object):
    def __init__(self, df):
        self.dim_feature = 5
        self.U = {}
        for index in df['user_id'].unique():
            self.U[index] = np.random.rand(self.dim_feature)
        self.T = {}
        for index in df['now_time'].unique():
            self.T[index] = np.random.rand(self.dim_feature)
        self.L1 = {}
        self.L2 = {}
        self.L3 = {}
        self.places = list(set(df['last_place'].unique()) |set(df['now_place'].unique()))
        for index in self.places:
            self.L1[index] = np.random.rand(self.dim_feature)
            self.L2[index] = np.random.rand(self.dim_feature)
            self.L3[index] = np.random.rand(self.dim_feature)
        
    
    def sigmoid(self,z):
        return 1.0 / (1.0+ math.exp(-z))
        
    def score(self,u,t,lq_2,lc_1,lc_2,lc_3,w):
        return lc_1.dot(u)+w*lc_2.dot(lq_2)+lc_3.dot(t)
            
    def train(self,df, reg=0.001, iterations = 100, k = 20, learning_rate = 0.0001):
        
        num_tuples = df.shape[0]
        
        places = np.array(self.places)
        
        for ite in xrange(iterations):
            
            loss = 0
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
                #print lnids
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
                    
                    loss_t += - math.log(self.sigmoid(fp-fn))  \
                    + 0.5 * reg * (np.sum(u**2)+np.sum(t**2)+np.sum(lp1**2) \
                    +np.sum(lp2**2)+np.sum(lp3**2) + np.sum(lq2**2) + np.sum(ln1**2) \
                    + np.sum(ln2**2) + np.sum(ln3**2))
                loss_t = loss_t / realk
                loss+=loss_t
                if i%1000 ==0 and i!=0:
                    loss = loss/1000
                    print "ite: %d/%d, tuple:%d/%d, loss: %s" %(ite,iterations,i,num_tuples,loss)
                    loss = 0
                    
    def test(self,df):
        num_tuples = df.shape[0]
        places = self.places
        correct_num=0
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
            u = self.U[u_id]
            t = self.T[t_id]
            
            lq2 = self.L2[lq_id]
            
            scores =[]
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
            print "now precision:", float(correct_num)/(i+1)
        precision = float(correct_num)/num_tuples
        print "precision: ", precision
        
model = stellar(df)
model.train(df)
model.test(df)


                
                
                
                
                
                
                