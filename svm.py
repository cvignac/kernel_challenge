# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:08:46 2018

@author: Raphael
"""

import numpy as np
from cvxopt import matrix, solvers

def signe(x):
    if x >= 0:
        return(1)
    else :
        return(0)

class svm :
    
    def __init__(self, y, G, lambd = 1 ):
        self.support_vectors = []
        n, d = np.shape(G)
        self.y = y.astype('double')*2 - np.ones(n)
        self.alpha = np.zeros(n)
        self.lambd = lambd
        self.K = G
                
    def calcule_alpha(self,lambd = -1):
        if lambd >= 0 :
            self.lambd = lambd
        n,d = np.shape(self.K)
        P = 2 * matrix(self.K,tc='d')
        q = -2 * matrix(self.y,(n,1))
        G = matrix(np.concatenate([np.diag(self.y),-np.diag(self.y)],0))
        h1 = np.ones(n) / (2*self.lambd*n)
        h2 = np.zeros(n) *0.
        h = matrix(np.concatenate([h1,h2],0),(2*n,1))
        sol = solvers.qp(P, q, G, h)
        self.alpha = np.array(sol['x'])
        
    def choix_support_vectors(self,seuil= 0.00000001):
        self.support_vectors = []
        for i in range(len(self.alpha)):
            if abs(self.alpha[i][0])>= seuil :
                self.support_vectors.append(i)
    
    def predict(self, G_):
        alpha = np.array([self.alpha[i] for i in self.support_vectors])
        P = np.dot(G_,alpha)
        prediction = np.array([signe(P[i]) for i in range(len(P))])
        return(prediction)
            
        
            