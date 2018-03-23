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
    
    def __init__(self,C):
        self.support_vectors = []
        self.y = np.zeros(1)
        self.alpha = np.zeros(1)
        self.C = C
        self.lambd = 1. /C
        self.K = np.zeros(1)
    
    def fit(self,G,y,):
        self.lambd = 1./self.C
        self.K = G
        n, d = np.shape(G)
        self.y = y.astype('double')*2 - np.ones(n)
        self.alpha = np.zeros(n)
        self.calcule_alpha()
        self.choix_support_vectors()
   
    def change_C(self,C):
        self.C = C
        self.lambd =1./C
                
    def calcule_alpha(self,C = -1):
        if C >= 0 :
            self.change_C(C)
        n,d = np.shape(self.K)
        P = 2 * matrix(self.K,tc='d')
        q = -2 * matrix(self.y,(n,1))
        G = matrix(np.concatenate([np.diag(self.y),-np.diag(self.y)],0))
        h1 = np.ones(n) / (2*self.lambd*n)
        h2 = np.zeros(n) *0.
        h = matrix(np.concatenate([h1,h2],0),(2*n,1))
        sol = solvers.qp(P, q, G, h)
        self.alpha = np.array(sol['x'])
        
    def choix_support_vectors(self,seuil= 0):
        self.support_vectors = []
        for i in range(len(self.alpha)):
            if abs(self.alpha[i][0])>= seuil :
                self.support_vectors.append(i)
    
    def predict(self, G_):
        alpha = np.array([self.alpha[i] for i in self.support_vectors])
        P = np.dot(G_,alpha)
        prediction = np.array([signe(P[i]) for i in range(len(P))])
        return(prediction)
            
        
            