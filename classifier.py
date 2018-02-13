import numpy as np
from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        try:
            return self.clf.predict(X)
        except AttributeError:
            return np.asarray(self.model.predict_proba(X) > 0.5, dtype=int)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def score(Y, Ytrue):
        ''' Accuracy metric.'''
        Y, Ytrue = Y.flatten(), Ytrue.flatten()
        assert len(Y) == len(Ytrue), 'len(Y)={} but len(Ytrue)={}'.format(
                len(Y), len(Ytrue))
        return np.sum(Y.flatten != Ytrue) / len(Y)


class MyClassifier(Classifier):
    def __init__(self):
        pass
