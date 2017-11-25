#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:49:21 2017

@author: nanthini, harshat
"""

import numpy as np
import random
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
def mykmeans(K, isRandom):
    '''
    
    Performs initialization of cluster centroids and mini-batch k-means
    
    Arguments:
     - K: integer defining the number of clusters
     - isRandom: boolean which tells us how to seed the algorithm; 
                 Randomized initial centrois if True; 
                 K++ is used otherwise
    
    '''
    k1 = K # Number of centers
    
    C = np.zeros((k1,np.shape(H)[1])) # Actual centers (matrix)
    if (isRandom):
        C = H[np.random.choice(np.arange(len(H)), K), :]
    else:
        '''
        
        Using K++ means to initialize centers by picking them 
        as far away from each other as possible. Here we pick one center
        at random and then find points that are far from the selected centers
        and repeat this process till we get all K centers!
        
        '''
        count = 1
        
        Chk = random.sample(H,1)
        C[0,:] = Chk[0]
        minvec = np.ones((np.shape(H)[0],))
        minvec = np.multiply(minvec,1000)

        for j in range(k1-1):
            C1 = np.matlib.repmat(Chk,np.shape(H)[0],1)
            C1 = C1 - H
            C1 = np.multiply(C1,C1)
            C2 = np.sum(C1,axis = 1)
            minvec = np.minimum(C2,minvec)
            csum = np.cumsum(minvec)
            maxVal = csum[np.shape(H)[0]-1]
            nextCenter = np.searchsorted(csum, np.random.random_sample()*maxVal)#, side = 'right') 
            Chk = H[nextCenter,:]
            C[count,:] = Chk
            count += 1
    print "Initialized Centers, now to find the optimal centers"
    

    count = [0 for i in range(K)]
    '''
    
    Performing mini-batch K-Means by randomly selecting 20000 points
    to adjust the centroids. This process is 250 times
    
    We stored the values of the distances between centers and their closest 
    data points for the generation of the plots!
    
    '''
    plottingmax = []
    plottingmin = []
    plottingavg = []
    for i in range(50): #Number of iterations is 50
    #    print i
        vals = []
        M = np.array(random.sample(H,10000))
        for sample in M:
            whereItBelongs = np.argmin([np.dot(sample-y_k, sample-y_k) for y_k in C])
            vals.append(min([np.dot(sample-y_k, sample-y_k) for y_k in C]))
            c = whereItBelongs
            count[c]+=1
            eta = 1./count[c]
            C[c] = (1.-eta)*C[c] + eta*sample
        plottingmax.append(max(vals))
        plottingmin.append(min(vals))
        plottingavg.append(np.mean(vals))
    return plottingmin, plottingmax, plottingavg


print "Reading the file..."
data = pd.read_csv('R6/ydata-fp-td-clicks-v1_0.20090501.gz',compression='gzip',
             header=None,delimiter="|",usecols = [1])
data = np.array(data)
H = np.empty([data.shape[0],6])
print "Generating the feature matrix.."
for i in range(data.shape[0]):
    s = data[i][0].split()
    for j in range(6):
        H[i,j] = s[1:][j].split(':')[1]

mi, av, ma = mykmeans(50, False) # Kmeans++
mi1, av1, ma1 = mykmeans(50, True) # Random
cmi = mi
cav = av
cma = ma
cmi1 = mi1
cma1 = ma1
cav1 = av1
count = 1
for i in range(9):
    print i
    mi, av, ma = mykmeans(50, False) # Kmeans++
    mi1, av1, ma1 = mykmeans(50, True) # Random
    #minimumvaluesR.append(mi1)
    #minimumvaluesK.append(mi)
    #maximumvaluesR.append(ma1)
    #maximumvaluesK.append(ma)
    #avevaluesR.append(av1)
    #avevaluesK.append(av)
    cmi += mi
    cav += av
    cma += ma
    cmi1 += mi1
    cma1 += ma1
    cav1 += av1
    count += 1

sum1a = np.sum(np.reshape(cma1,(10,50)), axis = 0)
suma = np.sum(np.reshape(cma,(10,50)), axis = 0)
plt.plot(range(50), sum1a, label = 'Random')
plt.plot(range(50), suma, label = 'kmeans++')
#plt.plot.legend("Random","Kmeans++")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Cumulative L2 norm")
plt.title("Maximum Distance")
plt.show()