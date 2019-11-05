# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 06:16:03 2019

@author: jairam
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:48:43 2019

@author: jairam
"""


import numpy as np
import sys
from math import sqrt
import csv
from sklearn.metrics import confusion_matrix

def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.8 * len(df))

    dataset_train = df[:split_1]

    dataset_test = df[split_1:]
    return dataset_train ,dataset_test


# =============================================================================

# =============================================================================

#output layer actiavtion fns


def softmax(x, beta,derivative=False):
    if (derivative == True):
        return beta*x * (1 - x)
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

#hiiden layer activation functions

def tanh_fn(x):
    return ( np.exp(x) - np.exp(-x)) / ( np.exp(x) + np.exp(-x))

def tanh(x,beta, derivative=False):
    if (derivative == True):
        return beta*(1 - (x ** 2))
    return tanh_fn(beta*x)

def softplus(x, beta,derivative=False):
    if(derivative==True):
        return 1 / (1 + np.exp(-x))
    return np.log(1+np.exp(x))

def elu(x,delta, derivative=False):
    if(derivative==True):
        return np.where(x>0,1,delta*np.exp(x))
    return np.where(x>0,x,delta*(np.exp(x)-1))
        
    

def relu(x,beta, derivative=False):
    if(derivative==True):
        return np.where(x>0,1,0)
    return np.where(x>0,x,0)

# =============================================================================
def sigmoid(x, beta, derivative=False):
    if (derivative == True):
        return beta*x * (1 - x)
    return 1 / (1 + np.exp(-beta*x))
# =============================================================================

    

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax, normalize = False):
    if normalize:
        for row in dataset:
            for i in range(len(row)-5):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
            





print('Enter 1 for Normalized or enter 0 for unnormalized. ')
s1 = input()

print('Enter the activation functions: \t 1.Logistic \t 2.Tanh \t 3.ReLU \t 4.Softplus \t 5.ELU')
s2 = input()

print('Enter learning mode \t 1.Pattern \t 2.Batch')
s3 = input()

print('Enter weight update rule \t 1.Delta \t 2.Generalized delta \t 3.AdaGrad \t 4.rmsprop \t 5.AdaDelta \t 6.Adam')
s4 = input()



X=list()            
with open('X.csv') as csvfile:
    readcsv=csv.reader(csvfile,delimiter=',')
    for row in readcsv:
        X.append([float(x) for x in row])
y=list()            
with open('y.csv') as csvfile:
    readcsv=csv.reader(csvfile,delimiter=',')
    for row in readcsv:
        y.append([float(y_) for y_ in row])
# =============================================================================
minmax = dataset_minmax(X)
normalize_dataset(X, minmax, bool(int(s1)))
# =============================================================================
np.random.seed(1)
X=np.asarray(X)   
y=np.asarray(y)
#shuffling dataset
dataset = np.concatenate((X,y),axis = 1)
np.random.shuffle(dataset)
df=dataset.tolist()
dataset_train,dataset_test=train_validate_test_split(df,0.7,0.1)
#train
dataset_train=np.asarray(dataset_train)
X_train = dataset_train[:,0:32]
y_train = dataset_train[:,32:]
#val

#test
dataset_test=np.asarray(dataset_test)
X_test = dataset_test[:,0:32]
y_test = dataset_test[:,32:]


def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])  #logp=N*1
    loss = np.sum(logp)  #computed for all examples(N)
    return loss

class MyNN:
    def __init__(self, x, y,neurons1,neurons2):
        self.x = x
        self.lr = 0.01
        ip_dim = x.shape[1]
        op_dim = y.shape[1]
        
        self.wIJ = np.random.randn(ip_dim, neurons1)#between input,first hidden(d*neurons1)
        self.bh1 = np.zeros((1, neurons1))#1*neurons1
# =============================================================================
        self.wJM = np.random.randn(neurons1, neurons2)#between 2nd hidden,first hidden(neurons1*n2)
        self.bh2 = np.zeros((1, neurons2))#1*n2
# =============================================================================
        self.wMK= np.random.randn(neurons2, op_dim)#between 2nd hidden,output(n2*K)
        self.bo = np.zeros((1, op_dim))#1*K
        self.y = y#N*K
        self.cs = dict()
        self.cs['normalize'] = dict()
        self.cs['normalize'] = {'True':True,'False':False}
        self.cs['activation function'] = dict()
        self.cs['activation function'] = {'Logistic':sigmoid, 'Tanh':tanh, 'ReLU':relu, 'Softplus':softplus, 'ELU':elu}
        self.cs['activation function derivative'] = {'Logistic':sigmoid, 'Tanh':tanh, 'ReLU':relu, 'Softplus':softplus, 'ELU':elu}
        self.cs['Learning mode'] = dict()
        #cs['Learning mode'] = {'Pattern':pattern(), 'Batch':batch()}
        self.cs['Pattern'] = dict()
        self.cs['Batch'] = dict()
        self.cs['Pattern']['feedforward'] = self.feedforward_pattern
        self.cs['Batch']['feedforward'] = self.feedforward_batch
        self.cs['Pattern']['Weight update'] = dict()
        self.cs['Batch']['Weight update'] = dict()        
        self.cs['Pattern']['Weight update'] = {'Delta':self.delta_pattern,'Generalised delta':self.gen_delta_pattern,'AdaGrad':self.adagrad_pattern,'rmsprop':self.rmsprop_pattern,'adadelta':self.adadelta_pattern,'adam':self.adam_pattern}
        self.cs['Batch']['Weight update'] = {'Delta':self.delta_batch}
        self.cs['Loss functions'] = dict()
        self.cs['lists'] = dict()
        self.cs['lists']={'rmsprop':[list(),list()],'adadelta':[[list(),list()],[list(),list()]]}
        
        
        #adadelta_pattern
        self.u_wmk = 0.00000001 + np.sqrt( np.zeros(self.wMK.shape))
        self.u_wjm = 0.00000001 + np.sqrt(np.zeros(self.wJM.shape))
        self.u_wij = 0.00000001 + np.sqrt( np.zeros(self.wIJ.shape))

        self.u_wbo = 0.00000001 + np.sqrt( np.zeros(self.bo.shape))
        self.u_wbh2 =0.00000001 +  np.sqrt( np.zeros(self.bh2.shape))
        self.u_wbh1 =0.00000001 +  np.sqrt( np.zeros(self.bh1.shape))
        
        self.adadelta_g = list()
        self.adadelta_b = list()
        self.adadelta_deltag = list()
        self.adadelta_deltab = list()
        
        #generalizeddelta_pattern
        self.delta_gwmk_prev = 0
        self.delta_gwjm_prev = 0
        self.delta_gwij_prev = 0
        self.delta_gbo_prev = 0
        self.delta_gbh2_prev = 0
        self.delta_gbh1_prev = 0
        
        #adam method
        self.adam_uwmk = np.zeros(self.wMK.shape)
        self.adam_uwjm = np.zeros(self.wJM.shape)
        self.adam_uwij = np.zeros(self.wIJ.shape)
        self.adam_ubo = np.zeros(self.bo.shape)
        self.adam_ubh2 = np.zeros(self.bh2.shape)
        self.adam_ubh1 = np.zeros(self.bh1.shape)
        
        self.adam_vwmk = np.zeros(self.wMK.shape)
        self.adam_vwjm = np.zeros(self.wJM.shape)
        self.adam_vwij = np.zeros(self.wIJ.shape)
        self.adam_vbo = np.zeros(self.bo.shape)
        self.adam_vbh2 = np.zeros(self.bh2.shape)
        self.adam_vbh1 = np.zeros(self.bh1.shape)
        
        
    def feedforward_pattern(self,n,beta):
        #for first hidden layer
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = self.cs['activation function'][s2](ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = self.cs['activation function'][s2](ah2,beta)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = self.cs['activation function']['Logistic'](ao,beta)#for output layer  #1*K#for output layer  #1*K
# =============================================================================
#         ah1 = np.dot(self.x, self.wij) + self.bh1#N*nuerons1
#         self.sh1 = sigmoid(ah1)#for first hidden layer
#         ah2 = np.dot(self.sh1, self.wjm) + self.bh2
#         self.sh2 = sigmoid(ah2)#N*nuerons2
#         ao = np.dot(self.sh2, self.wmk) + self.bo
#         self.so = softmax(ao)#for output layer  #N*K
# =============================================================================
        
#    def local_grad(self,node,layer):
        
    def feedforward_batch(self,rnum,beta):

        ah1 = np.dot(self.x, self.wIJ) + self.bh1
        self.sh1 = self.cs['activation function'][s2](ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = self.cs['activation function'][s2](ah2,beta)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = self.cs['activation function']["Logistic"](ao,beta)#for output layer  #1*K        
        
        

        
    def delta_pattern(self,n,beta):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * self.cs['activation function derivative']['Logistic'](self.so,beta,derivative=True)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * self.cs['activation function derivative'][s2](self.sh2,beta,derivative=True) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * self.cs['activation function derivative'][s2](self.sh1,beta,derivative=True) # w1
 
        self.wMK -= self.lr * np.dot(self.sh2.T, a3_delta)
        self.bo -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        self.wJM -= self.lr * np.dot(self.sh1.T, a2_delta)
        self.bh2 -= self.lr * np.sum(a2_delta, axis=0)
# =============================================================================
        self.wIJ -= self.lr * np.dot(x_n.T, a1_delta)
        self.bh1 -= self.lr * np.sum(a1_delta, axis=0)  
        
    
    def gen_delta_pattern(self,n,beta):
        
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * self.cs['activation function derivative']['Logistic'](self.so,beta,derivative=True)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * self.cs['activation function derivative'][s2](self.sh2,beta,derivative=True) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * self.cs['activation function derivative'][s2](self.sh1,beta,derivative=True) # w1
 
        gwMK =  np.dot(self.sh2.T, a3_delta)
        gbo =  np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        gwJM =  np.dot(self.sh1.T, a2_delta)
        gbh2 =  np.sum(a2_delta, axis=0)
# =============================================================================
        gwIJ =  np.dot(x_n.T, a1_delta)
        gbh1 =  np.sum(a1_delta, axis=0)
        
        if n==0:
            self.delta_gwmk_prev = 0
            self.delta_gwjm_prev = 0
            self.delta_gwij_prev = 0
            self.delta_gbo_prev = 0
            self.delta_gbh2_prev = 0
            self.delta_gbh1_prev = 0

        
        delta_gwmk = -1*self.lr*gwMK + 0.9*self.delta_gwmk_prev
        delta_gwjm = -1*self.lr*gwJM + 0.9*self.delta_gwjm_prev
        delta_gwij = -1*self.lr*gwIJ + 0.9*self.delta_gwij_prev
        
        delta_gbo = -1*self.lr*gbo + 0.9*self.delta_gbo_prev
        delta_gbh2 = -1*self.lr*gbh2 + 0.9*self.delta_gbh2_prev
        delta_gbh1 = -1*self.lr*gbh1 + 0.9*self.delta_gbh1_prev
    
        self.delta_gwmk_prev = delta_gwmk
        self.delta_gwjm_prev = delta_gwjm
        self.delta_gwij_prev = delta_gwij
        self.delta_gbo_prev = delta_gbo
        self.delta_gbh2_prev = delta_gbh2
        self.delta_gbh1_prev = delta_gbh1
        
        self.wMK += delta_gwmk
        self.bo += delta_gbo
# =============================================================================
        self.wJM += delta_gwjm
        self.bh2 += delta_gbh2
# =============================================================================
        self.wIJ += delta_gwij
        self.bh1 += delta_gbh1         

    
    def adagrad_pattern(self,n,beta):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * self.cs['activation function derivative']['Logistic'](self.so,beta,derivative=True)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * self.cs['activation function derivative'][s2](self.sh2,beta,derivative=True) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * self.cs['activation function derivative'][s2](self.sh1,beta,derivative=True) # w1
 
        gwMK =  np.dot(self.sh2.T, a3_delta)
        gbo =  np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        gwJM =  np.dot(self.sh1.T, a2_delta)
        gbh2 =  np.sum(a2_delta, axis=0)
# =============================================================================
        gwIJ =  np.dot(x_n.T, a1_delta)
        gbh1 =  np.sum(a1_delta, axis=0)  
        
        
        if n==0:
            stack = dict()
            stack[1] = list()
            stack[2] = list()
            stack[3] = list()
            
            stack1 = dict()
            stack1[1] = list()
            stack1[2] = list()
            stack1[3] = list()
            
            self.sum_gwmk = np.zeros(gwMK.shape)
            self.sum_gwjm = np.zeros(gwJM.shape)
            self.sum_gwij = np.zeros(gwIJ.shape)
            self.sum_gbo = np.zeros(gbo.shape)
            self.sum_gbh2 = np.zeros(gbh2.shape)
            self.sum_gbh1 = np.zeros(gbh1.shape)
            
        
        self.sum_gwmk += gwMK**2 
        self.sum_gwjm += gwJM**2 
        self.sum_gwij += gwIJ**2 
        
        self.sum_gbo  += gbo**2 
        self.sum_gbh2 += gbh2**2 
        self.sum_gbh1 += gbh1**2 
        
        
        r_wmk = 0.0000001 + np.sqrt(self.sum_gwmk)
        r_wjm = 0.0000001 + np.sqrt(self.sum_gwjm)
        r_wij = 0.0000001 + np.sqrt(self.sum_gwij)
        
        r_wbo  = 0.0000001 + np.sqrt(self.sum_gbo)
        r_wbh2 = 0.0000001 + np.sqrt(self.sum_gbh2)
        r_wbh1 = 0.0000001 +  np.sqrt(self.sum_gbh1)
        
        self.wMK -= self.lr * np.divide(gwMK,r_wmk)
        self.wJM -= self.lr * np.divide(gwJM,r_wjm)
        self.wIJ -= self.lr * np.divide(gwIJ,r_wij)        
        self.bo -= self.lr * np.divide(gbo,r_wbo)
        self.bh2 -= self.lr * np.divide(gbh2,r_wbh2)
        self.bh1 -= self.lr * np.divide(gbh1,r_wbh1)


    def rmsprop_pattern(self,n,beta):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * self.cs['activation function derivative']['Logistic'](self.so,beta,derivative=True)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * self.cs['activation function derivative'][s2](self.sh2,beta,derivative=True) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * self.cs['activation function derivative'][s2](self.sh1,beta,derivative=True) # w1
 
        gwMK =  np.dot(self.sh2.T, a3_delta)
        gbo =  np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        gwJM =  np.dot(self.sh1.T, a2_delta)
        gbh2 =  np.sum(a2_delta, axis=0)
# =============================================================================
        gwIJ =  np.dot(x_n.T, a1_delta)
        gbh1 =  np.sum(a1_delta, axis=0)
        
        gw = [gwIJ,gwJM,gwMK]
        gb = [gbh1,gbh2,gbo]
        
        self.cs['lists'][s4][0].append(gw)
        self.cs['lists'][s4][1].append(gb)
        if len(self.cs['lists'][s4][0])>10:
            self.cs['lists'][s4][0].pop(0)
        if len(self.cs['lists'][s4][1])>10:
            self.cs['lists'][s4][1].pop(0)
        
        r_wmk = 0.9*(np.sum(np.square([self.cs['lists'][s4][0][i][2] for i in range(len(self.cs['lists'][s4][0]))]),axis = 0)/10)+0.1*np.square(gwMK)        
        r_wjm = 0.9*(np.sum(np.square([self.cs['lists'][s4][0][i][1] for i in range(len(self.cs['lists'][s4][0]))]),axis = 0)/10)+0.1*np.square(gwJM) 
        r_wij = 0.9*(np.sum(np.square([self.cs['lists'][s4][0][i][0] for i in range(len(self.cs['lists'][s4][0]))]),axis = 0)/10)+0.1*np.square(gwIJ) 

        r_wbo = 0.9*(np.sum(np.square([self.cs['lists'][s4][1][i][2] for i in range(len(self.cs['lists'][s4][1]))]),axis = 0)/10)+0.1*np.square(gbo)   
        r_wbh2 = 0.9*(np.sum(np.square([self.cs['lists'][s4][1][i][1] for i in range(len(self.cs['lists'][s4][1]))]),axis = 0)/10)+0.1*np.square(gbh2)         
        r_wbh1 = 0.9*(np.sum(np.square([self.cs['lists'][s4][1][i][0] for i in range(len(self.cs['lists'][s4][1]))]),axis = 0)/10)+0.1*np.square(gbh1)         

        r_wmk =  np.sqrt(0.0000001 + r_wmk)
        r_wjm = np.sqrt(0.0000001 + r_wjm)
        r_wij =  np.sqrt(0.0000001 + r_wij)
        
        r_wbo  = np.sqrt(0.0000001 + r_wbo)
        r_wbh2 =  np.sqrt(0.0000001 + r_wbh2)
        r_wbh1 =  np.sqrt(0.0000001 + r_wbh1)


        self.wMK -= self.lr * np.divide(gwMK,r_wmk)
        self.wJM -= self.lr * np.divide(gwJM,r_wjm)
        self.wIJ -= self.lr * np.divide(gwIJ,r_wij)        
        self.bo -= self.lr * np.divide(gbo,r_wbo)
        self.bh2 -= self.lr * np.divide(gbh2,r_wbh2)
        self.bh1 -= self.lr * np.divide(gbh1,r_wbh1)


    def adadelta_pattern(self,n,beta):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * self.cs['activation function derivative']['Logistic'](self.so,beta,derivative=True)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * self.cs['activation function derivative'][s2](self.sh2,beta,derivative=True) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * self.cs['activation function derivative'][s2](self.sh1,beta,derivative=True) # w1
 
        gwMK =  np.dot(self.sh2.T, a3_delta)
        gbo =  np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        gwJM =  np.dot(self.sh1.T, a2_delta)
        gbh2 =  np.sum(a2_delta, axis=0)
# =============================================================================
        gwIJ =  np.dot(x_n.T, a1_delta)
        gbh1 =  np.sum(a1_delta, axis=0)
        
        gw = [gwIJ,gwJM,gwMK]
        gb = [gbh1,gbh2,gbo]
        
        self.adadelta_g.append(gw)
        self.adadelta_b.append(gb)
        if len(self.adadelta_g)>10:
            self.adadelta_g.pop(0)
        if len(self.adadelta_b)>10:
            self.adadelta_b.pop(0)
        
        r_wmk = 0.9*(np.sum(np.square([self.adadelta_g[i][2] for i in range(len(self.adadelta_g))]),axis = 0)/10)+0.1*np.square(gwMK)        
        r_wjm = 0.9*(np.sum(np.square([self.adadelta_g[i][1] for i in range(len(self.adadelta_g))]),axis = 0)/10)+0.1*np.square(gwJM) 
        r_wij = 0.9*(np.sum(np.square([self.adadelta_g[i][0] for i in range(len(self.adadelta_g))]),axis = 0)/10)+0.1*np.square(gwIJ) 

        r_wbo = 0.9*(np.sum(np.square([self.adadelta_b[i][2] for i in range(len(self.adadelta_b))]),axis = 0)/10)+0.1*np.square(gbo)   
        r_wbh2 = 0.9*(np.sum(np.square([self.adadelta_b[i][1] for i in range(len(self.adadelta_b))]),axis = 0)/10)+0.1*np.square(gbh2)         
        r_wbh1 = 0.9*(np.sum(np.square([self.adadelta_b[i][0] for i in range(len(self.adadelta_b))]),axis = 0)/10)+0.1*np.square(gbh1)         

        r_wmk = 0.00000001 + np.sqrt(r_wmk)
        r_wjm = 0.00000001 + np.sqrt(r_wjm)
        r_wij = 0.00000001 +  np.sqrt(r_wij)
        
        r_wbo  = 0.00000001 + np.sqrt(r_wbo)
        r_wbh2 = 0.00000001 +  np.sqrt(r_wbh2)
        r_wbh1 = 0.00000001 +  np.sqrt(r_wbh1)


        delta_wmk =np.multiply(self.u_wmk, np.divide(gwMK,r_wmk))
        delta_wjm =np.multiply(self.u_wjm, np.divide(gwJM,r_wjm))
        delta_wij =np.multiply(self.u_wij, np.divide(gwIJ,r_wij))
        
        delta_bo = np.multiply(self.u_wbo, np.divide(gbo,r_wbo))
        delta_bh2 = np.multiply(self.u_wbh2, np.divide(gbh2,r_wbh2))
        delta_bh1 = np.multiply(self.u_wbh1, np.divide(gbh1,r_wbh1))
        
        delta_g = [delta_wij,delta_wjm,delta_wmk]
        delta_b = [delta_bh1,delta_bh2,delta_bo]
        
        self.adadelta_deltag.append(delta_g)
        self.adadelta_deltab.append(delta_b)
        if len(self.adadelta_deltag)>10:
            self.adadelta_deltag.pop(0)
        if len(self.adadelta_deltab)>10:
            self.adadelta_deltab.pop(0)
        
        self.u_wmk = 0.00000001 + np.sqrt( 0.9*(np.sum(np.square([self.adadelta_deltag[i][2] for i in range(len(self.adadelta_deltag))]),axis = 0)/10)+0.1*np.square(delta_wmk))        
        self.u_wjm = 0.00000001 + np.sqrt(0.9*(np.sum(np.square([self.adadelta_deltag[i][1] for i in range(len(self.adadelta_deltag))]),axis = 0)/10)+0.1*np.square(delta_wjm)) 
        self.u_wij =0.00000001 + np.sqrt( 0.9*(np.sum(np.square([self.adadelta_deltag[i][0] for i in range(len(self.adadelta_deltag))]),axis = 0)/10)+0.1*np.square(delta_wij)) 

        self.u_wbo =0.00000001+np.sqrt( 0.9*(np.sum(np.square([self.adadelta_deltab[i][2] for i in range(len(self.adadelta_deltab))]),axis = 0)/10)+0.1*np.square(delta_bo))   
        self.u_wbh2 =0.00000001+np.sqrt( 0.9*(np.sum(np.square([self.adadelta_deltab[i][1] for i in range(len(self.adadelta_deltab))]),axis = 0)/10)+0.1*np.square(delta_bh2))         
        self.u_wbh1 =0.00000001+np.sqrt( 0.9*(np.sum(np.square([self.adadelta_deltab[i][0] for i in range(len(self.adadelta_deltab))]),axis = 0)/10)+0.1*np.square(delta_bh1))         

        self.wMK -= delta_wmk
        self.wJM -= delta_wjm
        self.wIJ -= delta_wij        
        self.bo -= delta_bo
        self.bh2 -= delta_bh2
        self.bh1 -= delta_bh1
        
    def adam_pattern(self,n,beta):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * self.cs['activation function derivative']['Logistic'](self.so,beta,derivative=True)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * self.cs['activation function derivative'][s2](self.sh2,beta,derivative=True) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * self.cs['activation function derivative'][s2](self.sh1,beta,derivative=True) # w1
 
        gwMK =  np.dot(self.sh2.T, a3_delta)
        gbo =  np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        gwJM =  np.dot(self.sh1.T, a2_delta)
        gbh2 =  np.sum(a2_delta, axis=0)
# =============================================================================
        gwIJ =  np.dot(x_n.T, a1_delta)
        gbh1 =  np.sum(a1_delta, axis=0)
        
        
        adam_uwmk = 0.9*self.adam_uwmk + 0.1*gwMK
        adam_uwjm = 0.9*self.adam_uwjm + 0.1*gwJM
        adam_uwij = 0.9*self.adam_uwij + 0.1*gwIJ
        
        adam_ubo = 0.9*self.adam_ubo + 0.1*gbo
        adam_ubh2 = 0.9*self.adam_ubh2 + 0.1*gbh2
        adam_ubh1 = 0.9*self.adam_ubh1 + 0.1*gbh1
        
        self.adam_uwmk = adam_uwmk
        self.adam_uwjm = adam_uwjm
        self.adam_uwij = adam_uwij
        self.adam_ubo = adam_ubo
        self.adam_ubh2 = adam_ubh2
        self.adam_ubh1 = adam_ubh1
        
        adam_uwmk = adam_uwmk/(1-(0.9**(n+1)))
        adam_uwjm = adam_uwjm/(1-(0.9**(n+1)))
        adam_uwij = adam_uwij/(1-(0.9**(n+1)))
        adam_ubo = adam_ubo/(1-(0.9**(n+1)))
        adam_ubh2 = adam_ubh2/(1-(0.9**(n+1)))
        adam_ubh1 = adam_ubh1/(1-(0.9**(n+1)))
        
        adam_vwmk = 0.999*self.adam_vwmk + 0.001*(gwMK**2)
        adam_vwjm = 0.999*self.adam_vwjm + 0.001*(gwJM**2)
        adam_vwij = 0.999*self.adam_vwij + 0.001*(gwIJ**2)
        
        adam_vbo = 0.999*self.adam_vbo + 0.001*(gbo**2)
        adam_vbh2 = 0.999*self.adam_vbh2 + 0.001*(gbh2**2)
        adam_vbh1 = 0.999*self.adam_vbh1 + 0.001*(gbh1**2)
        
        self.adam_vwmk = adam_vwmk
        self.adam_vwjm = adam_vwjm
        self.adam_vwij = adam_vwij
        self.adam_vbo = adam_vbo
        self.adam_vbh2 = adam_vbh2
        self.adam_vbh1 = adam_vbh1
        
        adam_vwmk = adam_vwmk/(1-(0.999**(n+1)))
        adam_vwjm = adam_vwjm/(1-(0.999**(n+1)))
        adam_vwij = adam_vwij/(1-(0.999**(n+1)))
        adam_vbo = adam_vbo/(1-(0.999**(n+1)))
        adam_vbh2 = adam_vbh2/(1-(0.999**(n+1)))
        adam_vbh1 = adam_vbh1/(1-(0.999**(n+1)))
        
        adam_vwmk = 0.00000001 + np.sqrt(adam_vwmk)
        adam_vwjm = 0.00000001 + np.sqrt(adam_vwjm)
        adam_vwij = 0.00000001 + np.sqrt(adam_vwij)
        adam_vbo = 0.00000001 + np.sqrt(adam_vbo)
        adam_vbh2 = 0.00000001 + np.sqrt(adam_vbh2)
        adam_vbh1 = 0.00000001 + np.sqrt(adam_vbh1)
        
        self.wMK -= self.lr * np.divide(adam_uwmk,adam_vwmk)
        self.wJM -= self.lr * np.divide(adam_uwjm,adam_vwjm)
        self.wIJ -= self.lr * np.divide(adam_uwij,adam_vwij)        
        self.bo -= self.lr * np.divide(adam_ubo,adam_vbo)
        self.bh2 -= self.lr * np.divide(adam_ubh2,adam_vbh2)
        self.bh1 -= self.lr * np.divide(adam_ubh1,adam_vbh1)
       


    def delta_batch(self,rnum,beta):
               
        z3_delta = (self.so - self.y)/self.x.shape[0] # w3
        a3_delta = z3_delta * self.cs['activation function derivative']['Logistic'](self.so,beta,derivative=True)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * self.cs['activation function derivative'][s2](self.sh2,beta,derivative = True) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * self.cs['activation function derivative'][s2](self.sh1,beta, derivative = True) # w1
 
        self.wMK -= self.lr * np.dot(self.sh2.T, a3_delta)
        
        #print(self.wMK[0][0])
        self.bo -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        self.wJM -= self.lr * np.dot(self.sh1.T, a2_delta)
        self.bh2 -= self.lr * np.sum(a2_delta, axis=0)
# =============================================================================
        self.wIJ -= self.lr * np.dot(self.x.T, a1_delta)
        self.bh1 -= self.lr * np.sum(a1_delta, axis=0)       






    def predict_feedforward(self,xx,yy,beta):
        x_n = xx[np.newaxis,:]
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = self.cs['activation function'][s2](ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = self.cs['activation function'][s2](ah2,beta)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = sigmoid(ao,beta)#for output layer  #1*K  
        rmse = (0.5*np.sum((yy-np.array(self.so))**2))
        y_pred=np.where(self.so>0.3,1,0)
#        true_cls = yy.argmax()
#        pred_cls = self.so.argmax()
        return (rmse, y_pred)
# =============================================================================
#     def predict(self, data):
#         self.x = data
#         self.predict_feedforward()
#         return self.so.argmax()
#     
#         
# =============================================================================
def get_acc(x, y,beta):
    acc=0
    p=0
    r=0
    f=0
    for xx,yy in zip(x, y):
        s = model.predict_feedforward(xx,yy,beta)[1]
        tp=0
        fp=0
        tn=0
        fn=0
# =============================================================================
        for j in range(s.shape[1]):
            if(yy[j] == 0 and s[0][j] == 0):
                tn=tn+1
            elif(yy[j] == 1 and s[0][j] == 1):
                tp=tp+1
            elif(yy[j] == 0 and s[0][j] == 1):
                fp=fp+1
            elif(yy[j] == 1 and s[0][j] == 0):
                fn=fn+1
# =============================================================================
#        tn, fp, fn, tp = confusion_matrix(yy, s[0]).ravel()
        p=p+tp/(tp+fp+0.00000001)
        r=r+tp/(tp+fn+0.00000001)
        f=f+((2*tp)/((2*tp) + fp + fn + 0.00000001)) 
        acc=acc+(tp+tn)/(tp+tn+fp+fn)
    p=p/x.shape[0]
    r=r/x.shape[0]
    f=f/x.shape[0]
    acc=acc/x.shape[0]
    return (p,r,f,acc)

best_beta=0
best_neurons=0	
min_val_rmse=sys.maxsize
#check for convergence

beta = 1
neurons1 = 42
neurons2 = 42


model = MyNN(X_train, np.array(y_train),neurons1,neurons2)
sum_prev_error=0
n_epochs=1
max_iters = 2500        
for j in range(max_iters):
    sum_error=0
    threshold=0.0001
    if s3 == 'Pattern':
        for n in range(X_train.shape[0]):  #pattern mode
            model.cs[s3]['feedforward'](n,beta)
            model.cs[s3]['Weight update'][s4](n,beta)
    if s3 == 'Batch':
        model.cs[s3]['feedforward'](n,beta)
        model.cs[s3]['Weight update'][s4](n,beta)
    for xx,yy in zip(X_train, np.array(y_train)):
        sum_error+= model.predict_feedforward(xx,yy,beta)[0]   
    #convergence criterion
    
    sum_error = sum_error/X_train.shape[0]
    print(sum_error)
    if(abs(sum_error-sum_prev_error)<=threshold ):
        print('parameters : beta={} and neurons1={} and neurons2 = {}'.format(beta,neurons1, neurons2))
        print('convergence has reached with difference of total error=',sum_error-sum_prev_error)
        print('no of epochs for convergence=',n_epochs)
        print("**********")
        break
    
    #print(abs(sum_error-sum_prev_error))
    sum_prev_error=sum_error
    n_epochs+=1
    

#            print(sum_error) 
    
if n_epochs>max_iters:
    print('no of epochs for convergence=',n_epochs)    
train_tpl = get_acc(X_train, y_train,beta)
print("Training P ={} Training R = {} Training F = {} ".format(train_tpl[0],train_tpl[1],train_tpl[2]))
test_tpl=get_acc(X_test, y_test,beta)
print("Test P ={} Test R = {} Test F = {} ".format(test_tpl[0],test_tpl[1],test_tpl[2]))
#print('Number of epochs =',n_epochs)
print("\n\n")  
print('-----------------------------------------------------------')


