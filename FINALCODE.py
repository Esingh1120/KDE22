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
d=2
mean = np.ones(d)

#Define Gaussian Class
class Gaussian:
    def __init__(self, covmat=(0.1)*np.diag(v=np.ones(len(mean)), k=0)):
        self.covmat = covmat        
    def evaluate(self, x, mean):
        cov = self.covmat
        inv_cov = np.linalg.inv(cov)
        if (np.linalg.det(2*np.pi*cov))**(-0.5) >= 0.0:
            return((2*np.pi*(np.linalg.det(cov)))**(-0.5)*np.exp((-0.5)*(x-mean)@inv_cov@(x-mean)))
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
    P0 += (1000)**(-1.0)*g.evaluate(x,m)
   
#print(samples)
print(P0)
print(len(mean))  #Check dim


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


#Compare Values Pi to P0
#Want to count all points that satify (Pi > P0) for integration; save other points for later

Accept = []   #Accept Pi >= P0
Reject = []   #Reject because Pi < P0

for i in range(len(pvalsamples)):
    Pi = pvalsamples[i]
    if (Pi >= P0):
        #print('Yeehaw!')
        Accept.append([i,Pi])
    else:
        #print('Nope lol, you tried')
        Reject.append([i,Pi])

print(len(Accept))
print(len(Reject))

#print(Accept)
#print(Reject)  


# In[7]:


#Which Pi < P0 to accept b/c boundary point? Redemption time!
# k==Nearest Neighbors, if nearest neighbor number is too low, won't count points that are maybe important

NN= int(input('How many nearest neighbors?'))

Redemption = []
Death = []

# Reject[i] = [index,Pval]
for i in range(len(Reject)):
    Re = Reject[i][1] #Pval
    if (Re*(len(samples)- NN) >= P0):
        Redemption.append(Reject[i][0]) #append index
    else:
        Death.append(Re)
        
print(len(Redemption), len(Death))
print(Redemption)


# In[8]:


#Redemption needs to be KDTree'd to find it's nearest nearest neighbors, but better than before
#What might be work is instead of adding the pval to an array, add the index of the points that pass of fail. 
#So if the 357th point fails, you can just call samples[357] to get the point again

pxiarray = np.zeros(len(Redemption))
for i in range(len(Redemption)):
    xi = samples[Redemption[i]]
    xiquery = kdtree.query(xi, k=len(samples))
    xiindex = xiquery[1]
    xidata = kdtree.data[xiindex]   #Points that are NN
    Pxi = (len(samples))**(-1.0)
    for j in range(5*NN):
        Pxi += (len(samples))**(-1.0)*g.evaluate(xi, xidata[j])

    pxiarray[i]=Pxi

print(pxiarray)     


# In[9]:


#Check same conditions as before
Accept1 = []
Reject1 =[]
for i in range(len(pxiarray)):
    pxi = pxiarray[i]
    if (pxi >= P0):
        Accept1.append([i,pxi])
    else:
        Reject1.append([i, pxi])

print(len(Accept1))
print(len(Reject1))


# In[10]:


#Out on the side


Redemption1 = []
Death1 = []

for i in range(len(Reject1)):
    Re1 = Reject1[i][1] #Pval
    if (Re1*(len(samples)- NN) >= P0):
        Redemption1.append(Reject1[i][0]) #append index
    else:
        Death1.append(Re1)
        
print(len(Redemption1), len(Death1))
print(Redemption1)

