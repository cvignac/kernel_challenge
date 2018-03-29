# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:50:51 2018

@author: Raphael
"""

import numpy as np
from math import sqrt


class PCA:
    def __init__(self, size):
        ''' Chooses the number of dimensions to consider.
        size (int) : number of dimensions.'''
        self.size = size
        self.vectors = np.zeros((size,size))
        self.eig = []

    def to_real(self, M): 
        '''Takes the real value of a matrix.'''
        n, m = np.shape(M)
        N = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                N[i, j] = M[i, j].real
        return(N)
    
    def norme(self,v):
        '''l2 norm of a vector'''
        return(sqrt(np.dot(v,v.T)))
    
    def normalise(self,v):
        '''Divides the vector by its norm.'''
        return(v/self.norme(v))

    def normalise_mat(self, V):
        '''Normalises the rows of the matrix.'''
        l = []
        for i, v in enumerate(V):
            l.append(self.normalise(v))
        return(np.array(l))

    def fit(self, X):
        '''Select the vectors of the pca.'''
        S = np.dot(X.T,X)
        self.eig = np.linalg.eig(S)
        self.vectors = self.eig[1][:self.size]
        self.vectors = self.to_real(self.vectors)
        self.vectors = self.normalise_mat(self.vectors)
        
    def features(self,X):
        '''Computes the features of a set of vectors X according to the pca vectors.'''
        return(np.dot(X,self.vectors.T))
        
    def variance_proportion(self):
        '''Variance explained by the pca.'''
        s = sum(self.eig[0])
        t = sum(self.eig[0][:self.size])
        print(t)
        return(t/s)

    def choose_size(self, pourcentage_variance=0.8):
        '''Set the variance explained by the pca.'''
        s = sum(self.eig[0])
        t = 0
        self.size = 0
        while t < pourcentage_variance * s :
            t += self.eig[0][self.size]
            self.size += 1
        return(self.size)



