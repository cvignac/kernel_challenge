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
        if self.sigma is None:
            print('Warning: no value was passed for sigma.',
                  'A default value is chosen')
            self.sigma = self.get_base(X1)
            print('Value chosen for sigma:', self.sigma)
        if X2 is None:
            K_condensed = squareform(pdist(X1, metric='euclidean'))
            return np.exp(- np.square(K_condensed) / (2 * self.sigma))
        else:
            K_condensed = cdist(X1, X2, metric='euclidean')
            return np.exp(- np.square(K_condensed) / (2 * self.sigma))
    def get_base(self, X):
        ''' Compute the average of the non-zero distances between all pairs
            of points. '''
        assert isinstance(X, np.ndarray), 'X should be a matrix'
        distances = pdist(X, 'euclidean')
        return np.mean(distances[np.nonzero(distances)])


if __name__ == '__main__':
    X = np.array([[0, 1, 2],
                  [10, 1, 2]])
    Y = np.array([[10, 1, 2]])
    ker = Gaussian(1)
    print(ker(X))
    print(ker(X, X))
    assert np.array_equal(ker(X), ker(X, X))
    print(ker(X, Y))
