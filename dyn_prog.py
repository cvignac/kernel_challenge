# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:21:42 2018

@author: Raphael
"""

import numpy as np

alphabet = ['A','C','T','G']

def base(p):
    S = ['']
    for i in range(p):
        S_ = []
        for s in S :
            for lettre in alphabet:
                S_.append(s+lettre)
        S = S_
    return(S)
    
class kernel :
    
    def __init__(self,s,t,i,lambd):
        self.s = s
        self.t = t
        self.i = i
        self.lambd = lambd
        self.K1 = np.zeros((len(s),len(t)))
        self.K_1 = np.zeros((len(s),len(t)))
        self.K = np.zeros(self.i)
        self.S = np.zeros((len(s),len(t)))
        
    def calcule_1(self):
        for i in range(len(self.s)):
            for j in range(len(self.t)):
                if self.s[i] == self.t[j] :
                    self.K1[i,j] = self.lambd**2
                    self.K[0] += self.K1[i,j]
        self.K_1 = self.K1.copy()
        self.K1 = np.zeros((len(self.s),len(self.t)))
        self.K[0] /= self.lambd**2
        
    def calcule(self,l):
        self.S[0:(l-1),:] = 0
        self.S[:,0:(l-1)] = 0
        for i in range(len(self.s)):
            for j in range(len(self.t)):
                self.S[i,j] = self.K_1[i,j]
                if i > 0 :
                    self.S[i,j] += self.S[(i-1),j] * self.lambd
                if j > 0 :
                    self.S[i,j] += self.S[i,(j-1)] * self.lambd
                if i > 0 and j > 0 :
                    self.S[i,j] -= self.lambd**2 * self.S[(i-1),(j-1)]
                if self.s[i] == self.t[j] and i > 0 and j > 0 :
                    self.K1[i,j] = self.lambd**2 * self.S[(i-1),(j-1)]
                    self.K[l-1] += self.K1[i,j]
        self.K[l-1] /= self.lambd**(2*l)
        self.K_1 = self.K1.copy()
        self.K1 = np.zeros((len(self.s),len(self.t)))
        
    def calcul_total(self):
        self.calcule_1()
        for l in range(2,self.i+1):
            self.calcule(l)

    def value(self):
        self.calcul_total()
        return(self.K[self.i-1])
        
def Gram_kernel(X,i,lambd,titre):
    n = len(X)
    G = np.zeros((n,n))
    for l, x1 in enumerate(X):
        print(l)
        for j, x2 in enumerate(X):   
            if l <= j :
                ker = kernel(x1,x2,i,lambd)
                G[l,j] = ker.value()
                G[j,l] = G[l,j]
                if j % 100 == 0 :
                    print(str(l)+'  '+str(j))
        np.savetxt(titre+'.txt', G)
    return(G)
    
def Product(X,Y,i,lambd,titre):
    n = len(X)
    d = len(Y)
    G = np.zeros((n,d))
    for l, x1 in enumerate(X):
        print(l)
        for j, x2 in enumerate(Y):   
            ker = kernel(x1,x2,i,lambd)
            G[l,j] = ker.value()
            if j % 100 == 0 :
                print(str(l)+'  '+str(j))
        np.savetxt(titre+'.txt', G)
    return(G)
            