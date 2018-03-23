import numpy as np
from abc import ABC, abstractmethod
import kernels
import feature_extractor as fe
from sklearn import svm


class Classifier(ABC):
    @abstractmethod
    def __init__(self, l, C, method, sigma):
        '''Abstract class for a classifier.
            l (int): size of the individual sequences considered.
            C (float): penalization parameter for the SVM
            method (str): kernel to use: 'linear' or 'gaussian'
            sigma (float): width of the gaussian kernel if method=='gaussian'.
        '''
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
        ''' Predict and compute the accuracy metric.'''
        Y = self.predict(Xte, dataset)
        Y, Yte = Y.flatten(), Yte.flatten()
        assert len(Y) == len(Yte), 'len(Y)={} but len(Ytrue)={}'.format(
                len(Y), len(Yte))
        return np.sum(Y == Yte) / len(Y)

    def build_kernel(self, method='linear', sigma=None):
        ''' method (str): 'gaussian' or 'linear'
            sigma (double): width of the gaussian kernel if method=='gaussian'.
        '''
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


class SubstringKernelSVM(Classifier):
    def __init__(self, l=4, lambd=0.6, C=1.0, method='linear', sigma=None):
        Classifier.__init__(self, l, C, method, sigma)
        self.extractor = fe.Substring(l, lambd)


class MultipleKernelClassifier(Classifier):
    def __init__(self, c1, c2, c3):
        ''' classifier: typically, Logistic Regression or SVM
            c1, c2, c3 (Classifier): kernels on each dataset. '''
        self.ker = [c1, c2, c3]

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
    print('Nothing to test')
