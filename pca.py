# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:50:51 2018

@author: Raphael
"""

import numpy as np
from math import sqrt

class PCA:
    
    def __init__(self,size):
        self.size = size
        self.vectors = np.zeros((size,size))
        
    def to_real(self,M):
        n,m = np.shape(M)
        N = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                N[i,j] = M[i,j].real
        return(N)
    
    def norme(self,v):
        return(sqrt(np.dot(v,v.T)))
    
    def normalise(v):
        return(v/self.norme(v))
        
    def normalise_mat(V):
        l = []
        for i,v in enumerate(V):
            l.append(self.normalise(v))
        return(np.array(l))  
            
    def fit(self, X):
        S = np.dot(X.T,X)
        eig = np.linalg.eig(S)
        self.vectors = eig[1][:self.size]
        self.vectors = to_real(self.vectors)
        self.vectors = normalise_mat(self.vectors)
        
    def features(self,X):
        return(np.dot(X,self.vectors.T))
        
        