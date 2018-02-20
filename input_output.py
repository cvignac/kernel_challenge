#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr

def process_submission_file(output_name, Ysub):
    ''' Save a csv file with the prediction and the correct format.
        Ysub: list of size 3, each one containing the predictions on 1 dataset.
    '''
    Y0, Y1, Y2 = Ysub
    total_l = len(Y0) + len(Y1) + len(Y2)
    Y = np.concatenate((Y0, Y1, Y2)).reshape(-1, 1)
    Y = np.concatenate((np.arange(total_l).reshape(-1, 1), Y), axis=1)
    np.savetxt(output_name, Y, '%i', ',', header='Id,Bound', comments='')


def load_X_full(filename):
    ''' Load the full dataset as a training set in order to submit
        the prediction.'''
    X = np.loadtxt(filename)
    return X

def load_Y_full(filename):
    ''' Load all the labels in order to train before submission.'''
    Y = np.loadtxt(filename, np.bool, skiprows=1, usecols=1, delimiter=',')
    return Y

def load_split(Xfile, Yfile, percent_test, seed=100):
    ''' Load files and split them into train and test examples.'''
    X = np.loadtxt(Xfile)
    Y = np.loadtxt(Yfile, np.bool, skiprows=1, usecols=1, delimiter=',')
    assert X.shape[0] == Y.shape[0]
    npr.seed(seed)
    p = npr.permutation(X.shape[0])
    X, Y = X[p], Y[p]
    cut = int(X.shape[0] * (1 - percent_test / 100))
    Xtr, Xte = X[: cut], X[cut:]
    Ytr, Yte = Y[: cut], Y[cut:]
    return Xtr, Xte, Ytr, Yte


if __name__ == '__main__':
#    load_full('./data/Xtr0_mat50.csv')
#    load_Y_full('./data/Ytr0.csv')
#    Xtr, Xte, Ytr, Yte = load_split('./data/Xtr0_mat50.csv', './data/Ytr0.csv', 15)
    Y = [np.array([1, 2]), np.array([2, 3]), np.array([4])]
    process_submission_file('test.csv', Y)