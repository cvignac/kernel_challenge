#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist, pdist, squareform

class Kernel(ABC):
    @abstractmethod
    def __call__(self, X1, X2=None):
        pass

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
    def __init__(self, l=3):
        self.l = l
        self.z = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __call__(self, X1, X2=None):
        if X2 is None:
            # X1_feat = self.build_features(X1)
            X1_feat = X1
            return X1_feat @ X1_feat.T
        else:
            # X1_feat = self.build_features(X1)
            # X2_feat = self.build_features(X2)
            X1_feat = X1
            X2_feat = X2
            return X1_feat @ X2_feat.T

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

if __name__ == '__main__':
    X = np.array([[0, 1, 2],
                  [10, 1, 2]])
    Y = np.array([[10, 1, 2]])
    ker = Gaussian(1)
    print(ker(X))
    print(ker(X, X))
    assert np.array_equal(ker(X), ker(X, X))
    print(ker(X, Y))
