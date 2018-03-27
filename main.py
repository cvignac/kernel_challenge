#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import classifier
import input_output
import grid_search
import numpy as np
import numpy.random as npr

submit = False

custom_features = True

percent_test = 15

prefix = './data/'
submit_file = 'Ysub.csv'

grid_s = False
# param_grid = {'C':[0.1, 0.05, 0.01, 0.005, 0.001]}
# param_grid = {'l':[4, 5, 6]}
param_grid = {'C': [.2, .5, .8, 1, 2, 5, 8]}

if submit:
    if custom_features:
        submission_files = ['Xte0.csv', 'Xte1.csv', 'Xte2.csv']
    else:
        submission_files = ['Xte0_mat50.csv', 'Xte1_mat50.csv',
                            'Xte2_mat50.csv']
    submission_files = ['./data/{}'.format(file) for file in submission_files]

if custom_features:
    train_files = ['Xtr0.csv', 'Xtr1.csv', 'Xtr2.csv']
    data_type = 'str'
else:
    train_files = ['Xtr0_mat50.csv', 'Xtr1_mat50.csv', 'Xtr2_mat50.csv']
    data_type = 'float'

label_files = ['Ytr0.csv', 'Ytr1.csv', 'Ytr2.csv']

train_files = ['./data/{}'.format(file) for file in train_files]
label_files = ['./data/{}'.format(file) for file in label_files]

if __name__ == '__main__':
    seed = 1984
    np.random.seed(seed)
#    Observations
#    sans PCA, sigma entre 1000 et 3000 donne les memes resultats
#    pour cross valider sigma, decommenter les lignes dans kernel.py
# C a été choisi
# Ça a l'air de marcher mieux sans pca
    k1 = classifier.FoldedKSpectrumKernelSVM(l=6, method='linear',
                                             C=1)
    k2 = classifier.FoldedKSpectrumKernelSVM(l=6, method='linear',
                                             C=1)
    k3 = classifier.FoldedKSpectrumKernelSVM(l=6, method='linear',
                                             C=0.5)
    clf = classifier.MultipleKernelClassifier(k1, k2, k3)

    if submit:
        Ysub = []
        for i in range(3):
            print('Loading datasets {}...'.format(i))
            Xtr = input_output.load_X_full(train_files[i], data_type)
            Ytr = input_output.load_Y_full(label_files[i])
            Xte = input_output.load_X_full(submission_files[i], data_type)

            print("Training ...")
            clf.fit(Xtr, Ytr, i)

            print('Predicting ...')
            Ysub.append(clf.predict(Xte, i))

        print('Processing the results ...')
        input_output.process_submission_file(submit_file, Ysub)
        print('Done.')

    else:
        accuracies = np.zeros(3)
        for i in range(3):
            print('Loadig datasets {}...'.format(i))
            Xtr, Xte, Ytr, Yte = input_output.load_split(train_files[i],
                                                         label_files[i],
                                                         percent_test,
                                                         data_type,
                                                         seed = seed)
            print("Training ...")
            if grid_s:
                grid_search.gridSearchCV(clf, Xtr, Ytr, i, param_grid,
                                         nfolds=3, verbose=1)
            else:
                clf.fit(Xtr, Ytr, i)

            accu = clf.score(Xtr, Ytr, i)
            accuracies[i] = accu
            print('Training accuracy:', accu * 100, '%')

            print('Predicting and Evaluating...')
            accu = clf.score(Xte, Yte, i)
            accuracies[i] = accu
            print('Accuracy:', accu * 100, '%')

        print('Done. Average accuracy:', np.mean(accuracies) * 100, '%')
