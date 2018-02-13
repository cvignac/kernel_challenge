#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import classifier
import input_output
import numpy as np


submit = False

custom_features = False

percent_test = 15

prefix = './data/'

if submit:
    if custom_features:
        submission_files = ['Xte0.csv', 'Xte1.csv', 'Xte2.csv']
    else:
        submission_files = ['Xte0_mat50.csv', 'Xte1_mat50.csv',
                            'Xte2_mat50.csv']
if custom_features:
    train_files = ['Xtr0.csv', 'Xtr1.csv', 'Xtr2.csv']
else:
    train_files = ['Xtr0_mat50.csv', 'Xtr1_mat50.csv', 'Xtr2_mat50.csv']

label_files = ['Ytr0.csv', 'Ytr1.csv', 'Ytr2.csv']


if __name__ == '__main__':
    np.random.seed('1984')

    clf = classifier.MyClassifier()

    if submit:
        Ysub = []
        for i in range(3):
            print('Loading datasets {}...'.format(i))
            Xtr = input_output.load_full(train_files[i])
            Ytr = input_output.load_full(label_files[i])
            Xte = input_output.load_full(submission_files[i])

            print("Training ...")
            clf.fit(Xtr, Ytr)

            print('Predicting ...')
            Ysub.append(clf.predict(Xte))

        print('Processing the results ...')
        input_output.process_submission_file(Ysub)
        print('Done.')

    else:
        accuracies = np.zeros(3)
        for i in range(3):
            print('Loading datasets {}...'.format(i))
            Xtr, Xte = input_output.load_split(train_files[i], percent_test)
            Ytr, Yte = input_output.load_split(label_files[i], percent_test)

            print("Training ...")
            clf.fit(Xtr, Ytr)

            print('Predicting ...')
            Ypred = clf.predict(Xte)

            print('Evaluating ...')
            accu = clf.score(Ypred, Yte)
            accuracies[i] = accu
            print('Accuracy:', accu * 100, '%')

        print('Done. Average accuracy:', np.mean(accuracies) * 100, '%')
