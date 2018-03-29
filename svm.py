# -*- coding: utf-8 -*-
import numpy as np
from cvxopt import matrix, solvers


class svm:
    def __init__(self, C):
        ''' Soft-SVM optimizer.
            C (float): regularization parameter. '''
        self.support_vectors = []
        self.y = None
        self.alpha = None
        self.C = C
        self.lambd = 1. / C
        self.K = None

    def fit(self, G, y):
        '''Fit the classifior.
         G (matrix) : Gram matrix of the features.
         y (vector) : labels.'''
        self.lambd = 1./self.C
        self.K = G
        n, d = np.shape(G)
        self.y = y.astype('double') * 2 - np.ones(n)
        self.calcule_alpha()
        self.choix_support_vectors()

    def calcule_alpha(self, C=-1):
        ''' Solves the QP.'''
        n, d = np.shape(self.K)
        P = 2 * matrix(self.K, tc='d')
        q = -2 * matrix(self.y, (n, 1))
        G = matrix(np.concatenate([np.diag(self.y), -np.diag(self.y)], 0))
        h1 = np.ones(n) / (2 * self.lambd * n)
        h2 = np.zeros(n, dtype=np.float)
        conca = np.concatenate([h1, h2], 0)
        h = matrix(conca, (2 * n, 1))
        sol = solvers.qp(P, q, G, h)
        self.alpha = np.array(sol['x']).flatten()

    def choix_support_vectors(self, seuil=0):
        ''' Selects the support vectors.
        seuil (float) : parameter that selects the vectors such that alpha > seuil.'''
        self.support_vectors = np.where(np.abs(self.alpha) >= seuil)

    def predict(self, G_):
        ''' Predicts the labels.
        G_ (matrix) : matrix of the kernel values of the features against the support vectors.'''
        alpha = self.alpha[self.support_vectors]
        P = np.dot(G_, alpha)
        prediction = (P >= 0).astype(np.int)
        return(prediction)
