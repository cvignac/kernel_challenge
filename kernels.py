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


if __name__ == '__main__':
    X = np.array([[0, 1, 2],
                  [10, 1, 2]])
    Y = np.array([[10, 1, 2]])
    ker = Gaussian(1)
    print(ker(X))
    print(ker(X, X))
    assert np.array_equal(ker(X), ker(X, X))
    print(ker(X, Y))