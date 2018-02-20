import numpy as np
import numpy.random as npr
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
        return np.asarray(self.predict_proba(X) > 0.5, dtype=np.bool)

    def score(Y, Ytrue):
        ''' Accuracy metric.'''
        Y, Ytrue = Y.flatten(), Ytrue.flatten()
        assert len(Y) == len(Ytrue), 'len(Y)={} but len(Ytrue)={}'.format(
                len(Y), len(Ytrue))
        return np.sum(Y == Ytrue) / len(Y)


class RandomClassifier(Classifier):
    def __init__(self):
        '''For testing purpose only.'''
        return

    def fit(self, X, y):
        return

    def predict_proba(self, X):
        n = X.shape[0]
        Y = npr.rand(n)
        return Y


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
    Y1 = np.array([True, True, False])
    Y2 = np.array([False, False, False])
    print(Classifier.score(Y1, Y2))
    pass
