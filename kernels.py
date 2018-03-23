#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist, pdist, squareform

class Kernel(ABC):
    @abstractmethod
    def __call__(self, X1, X2=None):
        pass


class Linear(Kernel):
    def __call__(self, X1, X2=None):
        if X2 is None:
            return X1 @ X1.T
        else:
            return X1 @ X2.T


class Gaussian(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X1, X2=None):
        if X2 is None:
            K_condensed = squareform(pdist(X1, metric='euclidean'))
            return np.exp(- np.square(K_condensed) / (2 * self.sigma))
        else:
            K_condensed = cdist(X1, X2, metric='euclidean')
            return np.exp(- np.square(K_condensed) / (2 * self.sigma))


class Spectral(Kernel):
    def __init__(self, l, method, sigma=None):
        ''' method (str): 'gaussian' or 'linear'. '''
        if method == 'linear':
            self.k = Linear()
        elif method == 'gaussian':
            assert sigma is not None, 'Gaussian kernel used but sigma=None'
            self.k = Gaussian(sigma)
        else:
            raise ValueError("Kernel '{}' not implemented".format(method))

        self.l = l
        self.z = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __call__(self, X1, X2=None):
        return self.k(X1, X2)
#        if X2 is None:
#            # X1_feat = self.build_features(X1)
#            X1_feat = X1
#            return X1_feat @ X1_feat.T
#        else:
#            # X1_feat = self.build_features(X1)
#            # X2_feat = self.build_features(X2)
#            X1_feat = X1
#            X2_feat = X2
#            return X1_feat @ X2_feat.T

    def build_features(self, X):
        X_feat = np.zeros((X.shape[0], 4**self.l))
        for i in range(X.shape[0]):
            X_feat[i] = self.spectral_features(X[i])
        return X_feat

    def spectral_features(self, x):
        n = len(x)
        phi_x = np.zeros(4**self.l)

        ind_x = list(map(lambda x: self.z[x], x))
        mult = [4**i for i in np.arange(self.l-1, -1, -1)]

        mapped_x = list(map(lambda i: sum([ind_x[i+j]*mult[j] for j in range(len(mult))]), range(n-self.l+1) ))

        for ind in mapped_x:
            phi_x[ind] += 1

        return(phi_x)


class FoldedKSpectrum(Kernel):
    def __init__(self, l, method, sigma=None):
        ''' method (str): 'gaussian' or 'linear'. '''
        if 'method' == 'linear':
            self.k = Linear()
        elif 'method' == 'gaussian':
            assert sigma is not None, 'Gaussian kernel used but sigma=None'
            self.k = Gaussian(sigma)
        else:
            raise ValueError("Kernel '{}' not implemented".format(method))
        self.l = l
        self.z = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.li = self.generate_lists()

    def __call__(self, X1, X2=None):
        return self.k(X1, X2)
#        if X2 is None:
#            # X1_feat = self.build_features(X1)
#            X1_feat = X1
#            return X1_feat @ X1_feat.T
#        else:
#            # X1_feat = self.build_features(X1)
#            # X2_feat = self.build_features(X2)
#            X1_feat = X1
#            X2_feat = X2
#            return X1_feat @ X2_feat.T

    # TODO Define functions:
    # - To handle a 0 in the list (sum over one position in the sequence)
    # - To handle all 0 in the list (sum over some positions in the sequence,
    #   using previous function sequentially over the contiguous k-mer rep.)

    # TODO Alternatively, define a process that handles all 0 in the list
    # at once (e.g. define a list of indices over which to sum, then shift?)

    def build_features(self, X):
        X_feat = np.zeros((X.shape[0], 4**self.l))
        for i in range(X.shape[0]):
            sf = self.spectral_features(X[i])
            cur_row = []
            for lis in self.li:
                cur_row += self.handle_zero_list(sf, lis)
            if (i==0):
                X_feat = np.array(cur_row).reshape(1,-1)
            else:
                X_feat = np.concatenate((X_feat, np.array(cur_row).reshape(1,-1)))
        return X_feat

    def spectral_features(self, x):
        n = len(x)
        phi_x = np.zeros(4**self.l)

        ind_x = list(map(lambda x: self.z[x], x))
        mult = [4**i for i in np.arange(self.l-1, -1, -1)]

        mapped_x = list(map(lambda i: sum([ind_x[i+j]*mult[j] for j in range(len(mult))]), range(n-self.l+1) ))

        for ind in mapped_x:
            phi_x[ind] += 1

        return(phi_x)

    def handle_zero_list(self, sf, li):
        # FIXME Double check
        # Initialize feature
        phi = []

        # Transform l into a sequence of ranks r0 and r1
        n = len(li)
        r0_t, r1_t = self.split_list_0_1(li)
        r0 = [4**(n-1-i) for i in r0_t]
        r1 = [4**(n-1-i) for i in r1_t]
        r0.reverse()
        r1.reverse()
        # r0r = r0.reverse()
        # r1r = r1.reverse()

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
        # print('Shifts', shiftl)

        for sh in shiftl:
            to_sum_over = list(map(lambda x: x+sh, basel))
            # print('to_sum_over', to_sum_over)
            # print(sf[to_sum_over])
            phi.append(np.sum(sf[to_sum_over]))

        return phi

    def generate_lists(self):
        l = self.l
        e = '{0:0'+str(l-1)+'b}'
        li = list(range(2**(l-1)))
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



if __name__ == '__main__':
    X = np.array([[0, 1, 2],
                  [10, 1, 2]])
    Y = np.array([[10, 1, 2]])
    ker = Gaussian(1)
    print(ker(X))
    print(ker(X, X))
    assert np.array_equal(ker(X), ker(X, X))
    print(ker(X, Y))
