#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import classifier
import input_output
import grid_search
import numpy as np

submit = True
use_substring_kernel = True  # use the substring kernel for dataset 2


custom_features = True  # If False, use builtin features

percent_test = 15   # Percent of the dataset used for testing

prefix = './data/'  # Path to data
submit_file = 'Ysub.csv'  # Output file

grid_s = False                                # Perform grid search?
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
    k1 = classifier.FoldedKSpectrumKernelSVM(l=6, method='linear', C=1)
    k2 = classifier.FoldedKSpectrumKernelSVM(l=6, method='linear', C=1)
    if use_substring_kernel:
        k3 = classifier.SumClassifier(dataset=2, l=6, C=0.5, method='linear')
    else:
        k3 = classifier.FoldedKSpectrumKernelSVM(l=6, method='linear', C=0.5)
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
            print('Loading datasets {}...'.format(i))
            Xtr, Xte, Ytr, Yte = input_output.load_split(train_files[i],
                                                         label_files[i],
                                                         percent_test,
                                                         data_type,
                                                         seed=seed)
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
