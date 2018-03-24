#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr

def gridSearchCV(clf, X, Y, dataset, param_grid, loss=None, score=None,
                 nfolds=3, verbose=0):
    ''' param_grid supports only one argument.
        clf (Classifier): classifier to use
        X, Y (ndarray, array): training set
        score (method)
        loss (method): the corresponding score is -loss
        if score and loss are None, the score function of clf is used
        param_grid (dict): name of the parameter, values to test
        verbose (int): if > 0, messages are printed. '''
    for key, values in param_grid.items():
        scores = np.zeros((nfolds, len(values)))
        # Shuffle the whole set
        n = X.shape[0]
        # X = X.reshape(-1, 1)        # CHANGED Problem specific, may not work in other settings
        assert X.shape[0] == Y.shape[0]
        # Y = Y.reshape(-1, 1)
        p = npr.permutation(n)
        X, Y = X[p], Y[p]
#        print('X.shape', X.shape)
#        print('Y.shape', Y.shape)
        # full = np.concatenate((X, Y), axis=1)
        # npr.shuffle(full)
        # X, Y = full[:, :-1], full[:, -1]
        binary = len(np.unique(Y)) <= 2
        for i in range(nfolds):
            # Create training and validation set
            start = int(i * n / nfolds)
            end = int((i+1) * n / nfolds) if i != nfolds - 1 else n
            Xte = X[start: end]
#            print(Xte.shape)
            Yte = Y[start: end]
            mask = np.ones(n, dtype=np.bool)
            mask[start: end] = 0
            Xtr = X[mask]
            Ytr = Y[mask]

            # Modify the classifier
            for j, param in enumerate(values):
                if dataset==None:
                    setattr(clf, key, param)
                else:
                    setattr(clf.ker[dataset], key, param)
#                    print((clf.ker[dataset]).C)
                # Predict
                if verbose > 0:
                    print('Training...')
                clf.fit(Xtr, Ytr, dataset)
                if verbose > 0:
                    print('Predicting...')
                if (loss is not None) or (score is not None):
                    if hasattr(clf, 'predict_proba') and binary:
                        pred = clf.predict_proba(Xte, dataset)
                    else:
                        print('Warning: predict was used',
                              'instead of predict_proba')
                        pred = clf.predict(Xte, dataset)
                    if loss is not None:
                        scores[i, j] = - loss(pred, Yte)
                    else:
                        scores[i, j] = score(Yte, pred)
                else:
                    scores[i, j] = clf.score(Xte, Yte, dataset)
        mean_score = np.mean(scores, axis=0)
        best = np.argmax(mean_score)
        if verbose > 0:
            print('All predictions done. Scores:')
            print(scores)
            print('Mean scores:')
            print(mean_score)
            print('Best parameter for ', key + ':', values[best])

        # Refit the classifier on the full dataset
        if dataset==None:
            setattr(clf, key, param)
        else:
            setattr(clf.ker[dataset], key, values[best])
        if verbose > 0:
            print('Reffiting...')
        clf.fit(X, Y, dataset)
        if verbose > 0:
            print('Cross validation completed.')


if __name__ == '__main__':
    # Scikit learn is used only for testing purpose
    classification = True

    if classification:
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        X, y = load_iris(return_X_y=True)
        acc = accuracy_score
        param_grid = {'C':[1e-6, 1e-3, 1, 1e3]}
        clf = LogisticRegression()
        gridSearchCV(clf, X, y, param_grid, score=acc, verbose=1)
    else:
        from sklearn.datasets import load_boston
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error
        X, y = load_boston(return_X_y=True)
        clf = Ridge(alpha = 1)
        param_grid = {'alpha': [0.1, 1, 10, 100]}
        gridSearchCV(clf, X, y, param_grid, loss=mse, nfolds=3, verbose=1)
