# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:51:33 2019

@author: jairam
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:48:43 2019

@author: jairam
"""


import numpy as np
#from random import seed
import csv

def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.7 * len(df))
    split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    dataset_val = df[split_1:split_2]
    dataset_test = df[split_2:]
    return dataset_train,dataset_val,dataset_test


def sigmoid(s,beta):
    return 1/(1 + np.exp(-beta*s))
# 
def sigmoid_derv(s,beta):
    return beta*s * (1 - s)
# =============================================================================

#output layer actiavtion fns


def softmax(x, beta,derivative=False):
    if (derivative == True):
        return beta*x * (1 - x)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
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
        if(x>0):
            return 1
        else :
            return delta*np.exp(x)
    if(x>0):
        return x
    else :
        return delta*(np.exp(x)-1)
    

def relu(x, derivative=False):
    if(derivative==True):
        if(x>0):
            return 1
        else :
            return 0
    if(x>0):
        return x
    else :
        return 0

# =============================================================================
# def sigmoid(x, derivative=False,beta):
#     if (derivative == True):
#         return beta*x * (1 - x)
#     return 1 / (1 + np.exp(-beta*x))
# =============================================================================

    

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-5):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
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
# minmax = dataset_minmax(X)
# normalize_dataset(X, minmax)
# =============================================================================
np.random.seed(1)
X=np.asarray(X)   
y=np.asarray(y)
#shuffling dataset
dataset = np.concatenate((X,y),axis = 1)
np.random.shuffle(dataset)
df=dataset.tolist()
dataset_train,dataset_val,dataset_test=train_validate_test_split(df,0.7,0.1)
#train
dataset_train=np.asarray(dataset_train)
X_train = dataset_train[:,0:32]
y_train = dataset_train[:,32:]
#val
dataset_val=np.asarray(dataset_val)
X_val = dataset_val[:,0:32]
y_val = dataset_val[:,32:]
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
    def __init__(self, x, y,neurons1,neurons2,seed_no):
        #seed(seed_no)
        self.x = x
        self.lr = 0.1
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

    def feedforward(self,n,beta):
        #for first hidden layer
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = sigmoid(ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = sigmoid(ah2,beta)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = sigmoid(ao,beta)#for output layer  #1*K
# =============================================================================

        
         
        
        
    def backprop_delta(self,n,beta):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * sigmoid_derv(self.so,beta)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * sigmoid_derv(self.sh2,beta) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * sigmoid_derv(self.sh1,beta) # w1
 
        self.wMK -= self.lr * np.dot(self.sh2.T, a3_delta)
        self.bo -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        self.wJM -= self.lr * np.dot(self.sh1.T, a2_delta)
        self.bh2 -= self.lr * np.sum(a2_delta, axis=0)
# =============================================================================
        self.wIJ -= self.lr * np.dot(x_n.T, a1_delta)
        self.bh1 -= self.lr * np.sum(a1_delta, axis=0)       
# =============================================================================


    def predict_feedforward(self,xx,yy,beta):
        x_n = xx[np.newaxis,:]
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = sigmoid(ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = sigmoid(ah2,beta)#1*nuerons2
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
        for j in range(s.shape[1]):
            if(yy[j] == 0 and s[0][j] == 0):
                tn=tn+1
            elif(yy[j] == 1 and s[0][j] == 1):
                tp=tp+1
            elif(yy[j] == 0 and s[0][j] == 1):
                fp=fp+1
            elif(yy[j] == 1 and s[0][j] == 0):
                fn=fn+1
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
max_val_acc=0
#check for convergence
# =============================================================================
for beta in [1,0.2]:
    for neurons1 in [56,42,28]:
        for neurons2 in [42,28,14]:
            seed_no=1
            model = MyNN(X_train, np.array(y_train),neurons1,neurons2,seed_no)
            sum_prev_error=0
            n_epochs=1     
            for i in range(3000):
                sum_error=0 
                threshold=0.00001
                for n in range(X_train.shape[0]):  #pattern mode
                    model.feedforward(n,beta)
                    model.backprop_delta(n,beta)
                for xx,yy in zip(X_train, np.array(y_train)):
                    sum_error+= model.predict_feedforward(xx,yy,beta)[0]     
                #convergence criterion
                sum_error=sum_error/X_train.shape[0]
               # print(sum_error)
                if(abs(sum_error-sum_prev_error)<=threshold):
                    print('parameters : beta={} and neurons1={} and neurons2={}'.format(beta,neurons1,neurons2))
                    print('convergence has reached with difference of total error=',sum_error-sum_prev_error)
                    print('no of epochs for convergence=',n_epochs)
                    print("**********")
                    break
                sum_prev_error=sum_error
                n_epochs+=1
               # print(sum_error)   
            train_acc=get_acc(X_train, y_train,beta)
            print("Training P = {} Training R = {} Training F = {} Training accu = {}".format(train_acc[0],train_acc[1],train_acc[2],train_acc[3]))
            val_acc=get_acc(X_val, y_val,beta)
            print("Validation P = {} Validation R = {} Validation F = {} Vaidation acc = {}".format(val_acc[0],val_acc[1],val_acc[2],val_acc[3]))
            test_acc = get_acc(X_test, y_test,beta)
            print("Testing P = {} Testing R = {} Testing F = {} Testing acc = {}".format(test_acc[0],test_acc[1],test_acc[2],test_acc[3]))
            if (max_val_acc<val_acc[3]):
                max_val_acc=val_acc[3]
                best_beta=beta
                best_neurons=(neurons1,neurons2)
            print("\n\n")  
            print('-----------------------------------------------------------')
print('best parameters of base model with two hidden layer are:beta={} and neurons1={} and nuerons2={}'.format(best_beta,best_neurons[0],best_neurons[1]))        
# =============================================================================
#
# =============================================================================
		



#tupl = get_acc(x_train, np.array(y_train))

#print("Confusion matrix :")
#print(tupl[1])

