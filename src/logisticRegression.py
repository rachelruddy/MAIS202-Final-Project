
from data import df

df.head()

import pandas as pd
import numpy as np
import seaborn as sns


#softmax function
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # Subtract row-wise max for stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


#each row represents a datapoint, each column represnts a class; a point represents the probability (0-1) that a datapoint is in that class; for row i: [e^z(i0)/e^z(i0)+e^z(i1)+e^z(i2), e^z(i1)/e^z(i0)+e^z(i1)+e^z(i2), e^z(i2)/e^z(i0)+e^z(i1)+e^z(i2)]


class logisticRegression:

    def __init__(self, learning_rate, max_iters, epsilon):
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # represents termination condition (smallest objective change)
        self.max_iters = max_iters  # maximum number of iterations of gradient descent

    def one_hot_encoded (self, y, num_classes):
        #converts labels to one-hot encoding form (N x K matrix)
        y = y.astype(int)
        Y_one_hot = np.zeros((y.shape[0], num_classes))
        Y_one_hot[np.arange(y.shape[0]), y] = 1 
        return Y_one_hot

    #train the weights
    def fit(self, x, y): 
        x = np.c_[np.ones(x.shape[0]), x] #add a column of 1's at the start of X for biases
        K = len(np.unique(y))   #number of classes (high, low, or mid risk)
        Y = self.one_hot_encoded(y,K)
        N, D = x.shape #x is datapoints by features matrix; N is number of women tested; D is number of features
       
        self.W = np.zeros((D,K))
        gradient = np.zeros_like(self.W) #intialize matrix of 0 for gradient CE loss 
        gradient_norm = np.inf 
        iteration_num = 0
        
        #gradient descent
        while gradient_norm>self.epsilon and iteration_num<self.max_iters: 
            Z  = np.dot(x, self.W)  #x⋅W  --> (N x K matrix)
            P = softmax(Z)  #computes softmax probabilities (N x K)
            
            #gradient of the CE loss function
            gradient = np.dot(x.T, (P-Y)) / N  #1/N * X(transpose)⋅(P-Y); where (P-Y) is ŷ-y   (D x k matrix)
            
            self.W = self.W - self.learning_rate*gradient
            iteration_num+=1
            gradient_norm=np.linalg.norm(gradient) #update gradient norm (scalar value representing gradient size)
        
        return self

    #predict output for given input
    def predict (self, x):
        x = np.c_[np.ones(x.shape[0]), x] #add a column of 1's at the start of x for biases
        Z = np.dot(x, self.W)
        P = softmax(Z)
        predictions = np.argmax(P, axis=1)
        return predictions.astype(int) #returns an array of predicted class labels of each point (by highest probability of each row/datapoint)
        



