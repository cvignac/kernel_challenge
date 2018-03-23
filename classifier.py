import numpy as np
import numpy.random as npr
from abc import ABC, abstractmethod
import kernels
import feature_extractor as fe
from sklearn import svm


class Classifier(ABC):
    @abstractmethod
    def __init__(self, l, C, method, sigma):
        self.l = l
        self.C = C
        self.sigma = sigma
        self.kernel = self.build_kernel(method, sigma)
        self.svm = svm.SVC(kernel=self.kernel, C=self.C)

    def fit(self, X, y, dataset=None):
        self.kernel.sigma = self.sigma
        self.svm.C = self.C
        self.extractor.l = self.l
        features = self.extractor.build_features(X)
        self.svm.fit(features, y)

    def predict_proba(self, X, dataset=None):
        pass

    def predict(self, X, _=None):
        features = self.extractor.build_features(X)
        return self.svm.predict(features)

    def score(self, Xte, Yte, dataset):
        ''' Accuracy metric.'''
        Y = self.predict(Xte, dataset)
        Y, Yte = Y.flatten(), Yte.flatten()
        assert len(Y) == len(Yte), 'len(Y)={} but len(Ytrue)={}'.format(
                len(Y), len(Yte))
        return np.sum(Y == Yte) / len(Y)

    def build_kernel(self, method='linear', sigma=None):
        ''' method (str): 'gaussian' or 'linear'. '''
        if method == 'linear':
            return kernels.Linear()
        elif method == 'gaussian':
            return kernels.Gaussian(sigma)
        else:
            raise ValueError("Kernel '{}' not implemented".format(method))


class SpectralKernelSVM(Classifier):
    def __init__(self, l=3, C=1.0, method='linear', sigma=None):
        Classifier.__init__(self, l, C, method, sigma)
        self.extractor = fe.Spectral(l=l)


class FoldedKSpectrumKernelSVM(Classifier):
    def __init__(self, l=3, C=1.0, method='linear', sigma=None):
        Classifier.__init__(self, l, C, method, sigma)
        self.extractor = fe.FoldedKSpectrum(self.l)


class MultipleKernelClassifier(Classifier):
    def __init__(self, k1, k2, k3):
        ''' classifier: typically, Logistic Regression or SVM
        k1, k2, k3 ('Classifier' objects): kernels on each dataset. '''
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
