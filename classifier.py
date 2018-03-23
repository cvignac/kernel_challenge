import numpy as np
import numpy.random as npr
from abc import ABC, abstractmethod
import kernels

from sklearn import svm


class Classifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y, dataset=None):
        pass

    def predict_proba(self, X, dataset=None):
        pass

    def predict(self, X, dataset=None):
        return np.asarray(self.predict_proba(X, dataset) > 0.5, dtype=np.bool)

    def score(self, Xte, Yte, dataset):
        ''' Accuracy metric.'''
        Y = self.predict(Xte, dataset)
        Y, Yte = Y.flatten(), Yte.flatten()
        assert len(Y) == len(Yte), 'len(Y)={} but len(Ytrue)={}'.format(
                len(Y), len(Yte))
        return np.sum(Y == Yte) / len(Y)


class RandomClassifier(Classifier):
    def __init__(self):
        '''For testing purpose only.'''
        return

    def fit(self, X, y, _=None):
        return

    def predict_proba(self, X, _=None):
        n = X.shape[0]
        Y = npr.rand(n)
        return Y


class SpectralKernelSVM(Classifier):
    def __init__(self, l=3, C=1.0):
        self.l = l
        self.C = C
        self.ker = kernels.Spectral(l=l)
        print('C',C)
        self.svm = svm.SVC(kernel=self.ker, C=self.C, verbose=False)

    def fit(self, X, y, _=None):
        self.svm.C = self.C
        self.ker.l = self.l
        self.svm.fit(self.ker.build_features(X), y)

    def predict(self, X, _=None):
        return self.svm.predict(self.ker.build_features(X))


class FoldedKSpectrumKernelSVM(Classifier):
    def __init__(self, l=3, C=1.0):
        self.l = l
        self.C = C
        self.ker = kernels.FoldedKSpectrum(l=self.l)
        self.svm = svm.SVC(kernel=self.ker, C=self.C)

    def fit(self, X, y, _=None):
        self.svm.C = self.C
        self.ker.l = self.l
        self.svm.fit(self.ker.build_features(X), y)

    def predict(self, X, _):
        return self.svm.predict(self.ker.build_features(X))


class MultipleKernelClassifier(Classifier):
    def __init__(self, k1, k2, k3):
        ''' classifier: typically, Logistic Regression or SVM
        k1, k2, k3 ('Kernel' objects): kernels on each dataset. '''
        self.ker = [k1, k2, k3]

    def fit(self, X, y, dataset):
        ''' dataset (int): between 0 and 2. '''
        return self.ker[dataset].fit(X, y)

    def predict_proba(self, X, dataset):
        assert hasattr(self.classifier, 'predict_proba')
        return self.ker[dataset].predict_probas(X)

    def predict(self, X, dataset):
        return self.ker[dataset].predict(X)

if __name__ == '__main__':
    # Tests go here
    print(1)
