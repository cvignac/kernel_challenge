#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:01:38 2018

@author: raphael
"""
import numpy as np
from input_output import *
from feature_extractor import *
from svm import *
from sklearn.model_selection import train_test_split
from kernels import *


percent_test = 15

l = 5

i = 2
C = 0.8
# Meilleurs paramètres l = 6, C = 3, 0.8 et 1.3 (respectivement pour chaque dataset)
train_files = ['Xtr0.csv', 'Xtr1.csv', 'Xtr2.csv']
data_type = 'str'
label_files = ['Ytr0.csv', 'Ytr1.csv', 'Ytr2.csv']
features_files = ['features_X0_p4_lambda0.6.txt', 'features_X2_p4_lambda0.6.txt','features_X0_p4_lambda0.6.txt'  ]
train_files = ['./data/{}'.format(file) for file in train_files]
label_files = ['./data/{}'.format(file) for file in label_files]
features_files = ['./substring/{}'.format(file) for file in features_files]

#%%
print('Loading files')

X = load_X_full(train_files[i],data_type)
f = np.loadtxt(features_files[i])
Y = load_Y_full(label_files[i])
#%%
print('Train test split')


n = len(X)
A = list(range(n))

ind_train, ind_test = train_test_split(A,test_size=percent_test/100)

Xtr = X[ind_train]
ftr = f[ind_train]
Ytr = Y[ind_train]
Xte = X[ind_test]
fte = f[ind_test]
Yte = Y[ind_test]

#%%
print('Computing kfloded features')

F = FoldedKSpectrum(l)
F_train = F.build_features(Xtr)
F_test = F.build_features(Xte)
#%%
print('Merging features')
features_train = np.concatenate((F_train,ftr),axis=1)
features_test = np.concatenate((F_test,fte),axis=1)
#%%
print('Computing kernels')


K = Linear()
G_train = K(features_train,features_train)
G_test = K(features_test,features_train)
#%%

print('Training svm')
clf = svm(C)
clf.fit(G_train,Ytr)

print('Predicting')
pr_tr = clf.predict(G_train)
pr_te = clf.predict(G_test)
print('train accuracy : '+str(1-sum(abs(pr_tr-Ytr))/len(Ytr)))
print('test accuracy : '+str(1-sum(abs(pr_te-Yte))/len(Yte)))


