import numpy as np
from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def predict(self, X):
        return np.asarray(selfpredict_proba(X) > 0.5, dtype=int)

    def score(Y, Ytrue):
        ''' Accuracy metric.'''
        Y, Ytrue = Y.flatten(), Ytrue.flatten()
        assert len(Y) == len(Ytrue), 'len(Y)={} but len(Ytrue)={}'.format(
                len(Y), len(Ytrue))
        return np.sum(Y.flatten != Ytrue) / len(Y)


class KLR(Classifier):
    def __init__(self, kernel, lamb=1):
        ''' Kernel Logistic regression.
        kernel (object): instance of the abstract class Kernel
        lamb (float): ridge parameter
        Ktr (np.ndarray): Kernel matrix for the training set
        '''
        self.lamb = lamb
        self.Ktr = None
        self.kernel = kernel

    def fit(self, X, y):
        self.K_tr = self.kernel(X)

    def predict():
        pass


if __name__ == '__main__':
    # Tests go here
