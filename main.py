#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import classifier
import input_output
import numpy as np


submit = True

custom_features = False

percent_test = 15

prefix = './data/'
submit_file = 'Ysub.csv'

if submit:
    if custom_features:
        submission_files = ['Xte0.csv', 'Xte1.csv', 'Xte2.csv']
    else:
        submission_files = ['Xte0_mat50.csv', 'Xte1_mat50.csv',
                            'Xte2_mat50.csv']
    submission_files = ['./data/{}'.format(file) for file in submission_files]

if custom_features:
    train_files = ['Xtr0.csv', 'Xtr1.csv', 'Xtr2.csv']
else:
    train_files = ['Xtr0_mat50.csv', 'Xtr1_mat50.csv', 'Xtr2_mat50.csv']

label_files = ['Ytr0.csv', 'Ytr1.csv', 'Ytr2.csv']

train_files = ['./data/{}'.format(file) for file in train_files]
label_files = ['./data/{}'.format(file) for file in label_files]

if __name__ == '__main__':
    np.random.seed(1984)

    clf = classifier.RandomClassifier()

    if submit:
        Ysub = []
        for i in range(3):
            print('Loading datasets {}...'.format(i))
            Xtr = input_output.load_X_full(train_files[i])
            Ytr = input_output.load_Y_full(label_files[i])
            Xte = input_output.load_X_full(submission_files[i])

            print("Training ...")
            clf.fit(Xtr, Ytr)

            print('Predicting ...')
            Ysub.append(clf.predict(Xte))

        print('Processing the results ...')
        input_output.process_submission_file(submit_file, Ysub)
        print('Done.')

    else:
        accuracies = np.zeros(3)
        for i in range(3):
            print('Loading datasets {}...'.format(i))
            Xtr, Xte, Ytr, Yte = input_output.load_split(train_files[i],
                                                         label_files[i],
                                                         percent_test)

            print("Training ...")
            clf.fit(Xtr, Ytr)

            print('Predicting ...')
            Ypred = clf.predict(Xte)

            print('Evaluating ...')
            accu = classifier.Classifier.score(Ypred, Yte)
            accuracies[i] = accu
            print('Accuracy:', accu * 100, '%')

        print('Done. Average accuracy:', np.mean(accuracies) * 100, '%')
