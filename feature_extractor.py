#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np

class FeatureExtractor(ABC):
    @abstractmethod
    def build_features(self, X):
        pass

class Spectral(FeatureExtractor):
    '''
    Standard Spectrum Kernel mapping described in class
    NB: We chose not to use a sparse representation of the features
    (computational complexity)
    '''
    def __init__(self, l):
        self.l = l
        self.z = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def build_features(self, X):
        # Build mapping for all data points
        X_feat = np.zeros((X.shape[0], 4**self.l))
        for i in range(X.shape[0]):
            X_feat[i] = self.spectral_features(X[i])
        return X_feat

    def spectral_features(self, x):
        # Build spectral features
        n = len(x)
        phi_x = np.zeros(4**self.l)

        ind_x = list(map(lambda x: self.z[x], x))
        mult = [4**i for i in np.arange(self.l-1, -1, -1)]

        mapped_x = list(map(lambda i: sum([ind_x[i+j]*mult[j] for j in range(len(mult))]), range(n-self.l+1) ))

        for ind in mapped_x:
            phi_x[ind] += 1

        return(phi_x)


class FoldedKSpectrum(FeatureExtractor):
    '''
    Folded k-Spectrum Kernel mapping as described by Elmas et al. (2017) in
    'The folded k-spectrum kernel: A machine learning approach to detecting
    transcription factor binding sites with gapped nucleotide dependencies'
    We follow their implementation, and therefore refer the reader to their
    paper for additional details
    NB: We do not use feature elimination here
    '''

    def __init__(self, l):
        self.l = l
        self.z = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # Generate binary representations of features - e.g. 111011
        self.li = self.generate_lists()


    def build_features(self, X):
        # Build all folded features based on the continuous spectral features
        X_feat = np.zeros((X.shape[0], 4**self.l))
        sf = []
        for i in range(X.shape[0]):
            sf.append(self.spectral_features(X[i]))
        X = np.vstack(sf)
        feats = [X]
        for lis in self.li:
            tmp = self.handle_zero_list(X, lis)
            feats += tmp
        X_feat = np.hstack(feats)
        return X_feat

    def spectral_features(self, x):
        # Build standard spectral features
        n = len(x)
        phi_x = np.zeros(4**self.l)

        ind_x = list(map(lambda x: self.z[x], x))
        mult = [4**i for i in np.arange(self.l-1, -1, -1)]

        mapped_x = list(map(lambda i: sum([ind_x[i+j]*mult[j] for j in range(len(mult))]), range(n-self.l+1) ))

        for ind in mapped_x:
            phi_x[ind] += 1

        return(phi_x)

    def handle_zero_list(self, sf, li):
        # Build the features corresponding to a specific binary representation
        # li of dependent and independent nucleotides.

        # Initialize feature
        phi = []

        # Transform l into a sequence of ranks r0 and r1
        n = len(li)
        r0_t, r1_t = self.split_list_0_1(li)
        r0 = [4**(n-1-i) for i in r0_t]
        r1 = [4**(n-1-i) for i in r1_t]
        r0.reverse()
        r1.reverse()

        if len(r0)==0:
            return sf.tolist()

        # Define the base indices over which to sum for the first feature,
        # based on r0 (e.g. NAN)
        basel = [0]
        for rk in r0:
            basel2 = []
            for j in range(4):
                basel2 += list(map(lambda x: x+j*rk, basel))
            basel = basel2.copy()

        # Loop over the shifts in the previous baselien induced by r1
        # (e.g. NAN, NCN, NGN, NTN) and build the features
        # Be careful with the order of features
        shiftl = [0]
        for rk in r1:
            shiftl2 = []
            for j in range(4):
                shiftl2 += list(map(lambda x: x+j*rk, shiftl))
            shiftl = shiftl2.copy()

        # print('Base', basel)
        if (shiftl==[]): shiftl=[0]
        # print('Shifts', shiftl)

        for sh in shiftl:
            to_sum_over = list(map(lambda x: x+sh, basel))
            # print('to_sum_over', to_sum_over)
            # print(sf[to_sum_over])
            phi.append(np.sum(sf[:,to_sum_over], axis=1).reshape(-1,1))

        return phi

    def generate_lists(self):
        # Generate binary representations of features - e.g. 111011
        l = self.l
        e = '{0:0'+str(l-1)+'b}'
        li = list(range(2**(l-1) -1))
        li = list(map(lambda x: e.format(x), li))
        li = list(map(lambda x: x+'1', li))
        li.reverse()
        return li


    def split_list_0_1(self, li):
        r0, r1 = [], []
        for i in range(len(li)):
            if (li[i]=='0'):
                r0.append(i)
            if (li[i]=='1'):
                r1.append(i)
        return r0,r1


class Substring(FeatureExtractor):
    '''Substring kernel computed with dynamic programming.'''
    def __init__(self,l,lambd,s='',t=''):
        ''' l (int) : length of the substrings.
        lambd (float) : penalization parameter.'''
        self.s = s
        self.t = t
        self.l = l
        self.lambd = lambd
        self.K1 = np.zeros((len(s),len(t)))
        self.K_1 = np.zeros((len(s),len(t)))
        self.K = np.zeros(self.l)
        self.S = np.zeros((len(s),len(t)))
        self.alphabet = ['A','C','T','G']

    def base(self):
        S = ['']
        for j in range(self.l):
            S_ = []
            for s in S :
                for lettre in self.alphabet:
                    S_.append(s+lettre)
            S = S_
        return(S)

    def reinitialise(self):
        self.K1 = np.zeros((len(self.s),len(self.t)))
        self.K_1 = np.zeros((len(self.s),len(self.t)))
        self.K = np.zeros(self.l)
        self.S = np.zeros((len(self.s),len(self.t)))

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
        for l in range(2,self.l+1):
            self.calcule(l)

    def value(self):
        self.calcul_total()
        return(self.K[self.l-1])

    def Gram_kernel(self,X):
        n = len(X)
        G = np.zeros((n,n))
        for l, x1 in enumerate(X):
            self.s = x1
            for j, x2 in enumerate(X):
                if l <= j :
                    self.t = x2
                    self.reinitialise()
                    G[l,j] = self.value()
                    G[j,l] = G[l,j]
        return(G)

    def calcule_feature(self,s,S=[]):
        if S == []:
            S = self.base()
        l_s = []
        self.s = s
        for b in S :
            self.t = b
            self.reinitialise()
            l_s.append(self.value())
        return(l_s)

    def build_features(self, X):
        S = self.base()
        l = []
        for i,x in enumerate(X):
            l.append(self.calcule_feature(x,S))
        l = np.array(l)
        return(l)

class Substring_from_files(FeatureExtractor):
    ''' Load the features in the files of the folder substring.'''

    def __init__(self,dataset):
        self.dataset = dataset

    def build_features(self,X):
        if len(X) == 2000 :
            features_files = ['features_X0_p4_lambda0.6.txt', 'features_X2_p4_lambda0.6.txt','features_X0_p4_lambda0.6.txt'  ]
            features_files = ['./substring/{}'.format(file) for file in features_files]
            return(np.loadtxt(features_files[self.dataset]))
        elif len(X) == 1000 :
            features_files = ['features_test0_p4_lambda0.6.txt', 'features_test1_p4_lambda0.6.txt','features_test2_p4_lambda0.6.txt'  ]
            features_files = ['./substring/{}'.format(file) for file in features_files]
            return(np.loadtxt(features_files[self.dataset]))
        else :
            print('Substring from files can only be called by the original datasets.')
            return(np.zeros((len(X),1)))

class Sum(FeatureExtractor):

    def __init__(self,F1,F2):
        self.F1 = F1
        self.F2 = F2

    def build_features(self,X):
        return(np.concatenate((self.F1.build_features(X),self.F2.build_features(X)),axis=1))
