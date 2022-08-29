#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reference
import numpy as np
import scipy as sp
from scipy.spatial import KDTree
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import math

import os
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


#Sampling from Guassian => Find P(0) (using KDE); The Model of 0 Vector
#d==dimension
d=4
mean = np.ones(d)

#Define Gaussian Class
class Gaussian:
    def __init__(self, covmat=(0.5)*np.diag(v=np.ones(len(mean)), k=0)):
        self.covmat = covmat        
    def evaluate(self, x, mean):
        cov = self.covmat
        inv_cov = np.linalg.inv(cov)
        if (np.linalg.det(2*np.pi*cov))**(-0.5) >= 0.0:
            return((2*np.pi*(np.linalg.det(cov)))**(-0.5) * np.exp(-0.5*(x-mean)@inv_cov@(x-mean)))
        else:
            print("Determinant of covariance matrix is not positive definite.")

#Sample Points from Gaussian
g = Gaussian()
mean = np.ones(d)
x=np.zeros(len(mean))
covariance = np.diag(v=np.ones(len(mean)), k=0)
samples = np.random.multivariate_normal(mean, covariance, size=1000)
P0 = 0
for m in samples:
    P0 += (1000)**(-1.0)*g.evaluate(x,mean)
   
print(samples)
print(P0)
print(len(mean))  #Check dim

'''Changing value of covmat class, aka smoothing parameter, is main control for P(0) value being so low. \n
    When I increase smoothing parameter, P(0) is higher (as expected), but 0.5 seems to be happy smoothing lol'''


# In[3]:


#Check for P(0) analytically 


# In[4]:


#Generate Random Samples, Used as a check
sample_length = [10, 20, 30, 40, 50, 75, 100, 1000]
pval = np.zeros(len(sample_length))
for i in range(len(sample_length)):
    length = sample_length[i]
    samples = np.random.multivariate_normal(mean, covariance, size=length)
    P=0
    for m in samples:
        P += (length)**(-1.0)*g.evaluate(x,m)
    
    pval[i] = P
    print(pval)

plt.plot(sample_length, pval)
plt.xscale('log')


# In[5]:


#KD-Tree Time :O
#From sample lengths, find k+1 nearest neighbors with kdtree, k=100 nearest neighbors
kdtree = KDTree(samples)
query = kdtree.query(x=np.zeros(d), k=100)
#Query KD-Tree for nearest Neighbors
print(query)
index = query[1]
print(index)
data = kdtree.data[index]
#Get data points of the nearest neighbors
print(data)


# In[6]:


#Gaussian Around Sampled Points

pvalsamples = np.zeros(len(samples))

#Pick one data point as modelled point, compare every other point to "modelled" point
for i in range(len(samples)):
    modpoint = samples[i]
    modquery = kdtree.query(modpoint, k=100)
    modindex = modquery[1]
    moddata = kdtree.data[modindex]
    print('\r' +str(i),end='')
    Pd = (1/1000)
    for j in range(len(moddata)):
        Pd += (1000)**(-1.0)*g.evaluate(moddata[j], modpoint)
        
    pvalsamples[i] = Pd #Pval of samples 

#print(len(data))
#print(data)

print(pvalsamples)


# In[7]:


#Compare Values Pi to P0
#Want to count all points that satify (Pi > P0) for integration; save other points for later

Accept = []   #Accept Pi >= P0
Reject = []   #Reject because Pi < P0

for i in range(len(pvalsamples)):
    Pi = pvalsamples[i]
    if (Pi >= P0):
        print('Yeehaw!')
        Accept.append(Pi)
    else:
        print('Nope lol, you tried')
        Reject.append(Pi)

print(len(Accept))
print(len(Reject))

print(Accept)
print(Reject)  


# In[8]:


#Which Pi < P0 to accept b/c boundary point? Redemption time!
# k==Nearest Neighbors, if nearest neighbor number is too low, won't count points that are maybe important

NN= int(input('How many nearest neighbors?'))

Redemption = []
Death = []

for i in range(len(Reject)):
    Re = Reject[i]
    if (Re*(len(samples)- NN) >= P0):
        Redemption.append(Re)
    else:
        Death.append(Re)
        
print(len(Redemption), len(Death))
print(Redemption)


# In[23]:


#Redemption needs to be KDTree'd to find it's nearest nearest neighbors, but better than before

redemption = np.zeros(len(Redemption))

for i in range(len(samples)):
    re = Redemption[i]
    requery = kdtree.query(re, k=5*NN)
    reindex = requery[1]
    redata = kdtree.data[reindex]
    print('\r' +str(i),end='')
    reP = (1/1000)
    for j in range(len(redata)):
        reP += (1000)**(-1.0)*g.evaluate(redata[j], re)
        

redemption[i] = reP

