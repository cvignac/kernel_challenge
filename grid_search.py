#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr

def gridSearchCV(clf, X, Y, param_grid, loss=None, nfolds=3, verbose=0):
    ''' param_grid supports only one argument.
        clf (Classifier): classifier to use
        X, Y (ndarray, array): training set
        loss (method): if None, the score function of clf is used
                       else, the score used is -loss
        param_grid (dict): name of the parameter, values to test
        verbose (int): if > 0, messages are printed. '''
    for key, values in param_grid.items():
        scores = np.zeros((nfolds, len(values)))
        # Shuffle the whole set
        n = X.shape[0]
        assert X.shape[0] == Y.shape[0]
        Y = Y.reshape(-1, 1)
        full = np.concatenate((X, Y), axis=1)
        npr.shuffle(full)
        X, Y = full[:, :-1], full[:, -1]

        for i in range(nfolds):
            # Create training and validation set
            start = int(i * n / nfolds)
            end = int((i+1) * n / nfolds) if i != nfolds - 1 else n
            Xte = X[start: end]
            Yte = Y[start: end]
            mask = np.ones(n, dtype=np.int)
            mask[start: end] = 0
            Xtr = X[mask]
            Ytr = Y[mask]

            # Modify the classifier
            for j, param in enumerate(values):
                setattr(clf, key, param)
                # Predict
                if verbose > 0:
                    print('Training...')
                clf.fit(Xtr, Ytr)
                if verbose > 0:
                    print('Predicting...')
                if hasattr(clf, 'predict_proba'):
                    pred = clf.predict_proba(Xte)
                else:
                    print('Warning: predict was used instead of predict_proba')
                    pred = clf.predict(Xte)
                if loss is None:
                    scores[i, j] = clf.score(pred.reshape(-1, 1),
                                             Yte.reshape(-1, 1))
                else:
                    scores[i, j] =  - loss(pred, Yte)
        mean_score = np.mean(scores, axis=0)
        best = np.argmax(mean_score)
        if verbose > 0:
            print('All predictions done. Scores:')
            print(scores)
            print('Mean scores:')
            print(mean_score)
            print('Best parameter for ', key + ':', values[best])

        # Refit the classifier on the full dataset
        setattr(clf, key, values[best])
        if verbose > 0:
            print('Reffiting...')
        clf.fit(X, Y)
        if verbose > 0:
            print('Cross validation completed.')


if __name__ == '__main__':
    # Scikit learn is used only for testing purpose
    from sklearn.datasets import load_boston
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error
    X, y = load_boston(return_X_y=True)
    print(X.shape, y.shape)
    clf = Ridge(alpha = 1)
    param_grid = {'alpha': [0.1, 1, 10, 100]}
    gridSearchCV(clf, X, y, param_grid, loss=mse, nfolds=3, verbose=1)