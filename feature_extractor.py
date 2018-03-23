#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def build_features(self, X):
        pass

class Spectral(FeatureExtractor):
    def __init__(self, l, method, sigma=None):
        ''' method (str): 'gaussian' or 'linear'. '''
        self.l = l
        self.z = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def build_features(self, X):
        X_feat = np.zeros((X.shape[0], 4**self.l))
        for i in range(X.shape[0]):
            X_feat[i] = self.spectral_features(X[i])
        return X_feat

    def spectral_features(self, x):
        n = len(x)
        phi_x = np.zeros(4**self.l)

        ind_x = list(map(lambda x: self.z[x], x))
        mult = [4**i for i in np.arange(self.l-1, -1, -1)]

        mapped_x = list(map(lambda i: sum([ind_x[i+j]*mult[j] for j in range(len(mult))]), range(n-self.l+1) ))

        for ind in mapped_x:
            phi_x[ind] += 1

        return(phi_x)


class FoldedKSpectrum(FeatureExtractor):
    def __init__(self, l, method, sigma=None):
        self.l = l
        self.z = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.li = self.generate_lists()
    # TODO Define functions:
    # - To handle a 0 in the list (sum over one position in the sequence)
    # - To handle all 0 in the list (sum over some positions in the sequence,
    #   using previous function sequentially over the contiguous k-mer rep.)

    # TODO Alternatively, define a process that handles all 0 in the list
    # at once (e.g. define a list of indices over which to sum, then shift?)

    def build_features(self, X):
        X_feat = np.zeros((X.shape[0], 4**self.l))
        for i in range(X.shape[0]):
            sf = self.spectral_features(X[i])
            cur_row = []
            for lis in self.li:
                cur_row += self.handle_zero_list(sf, lis)
            if (i==0):
                X_feat = np.array(cur_row).reshape(1,-1)
            else:
                X_feat = np.concatenate((X_feat, np.array(cur_row).reshape(1,-1)))
        return X_feat

    def spectral_features(self, x):
        n = len(x)
        phi_x = np.zeros(4**self.l)

        ind_x = list(map(lambda x: self.z[x], x))
        mult = [4**i for i in np.arange(self.l-1, -1, -1)]

        mapped_x = list(map(lambda i: sum([ind_x[i+j]*mult[j] for j in range(len(mult))]), range(n-self.l+1) ))

        for ind in mapped_x:
            phi_x[ind] += 1

        return(phi_x)

    def handle_zero_list(self, sf, li):
        # FIXME Double check
        # Initialize feature
        phi = []

        # Transform l into a sequence of ranks r0 and r1
        n = len(li)
        r0_t, r1_t = self.split_list_0_1(li)
        r0 = [4**(n-1-i) for i in r0_t]
        r1 = [4**(n-1-i) for i in r1_t]
        r0.reverse()
        r1.reverse()
        # r0r = r0.reverse()
        # r1r = r1.reverse()

        if len(r0)==0:
            return sf.tolist()

        # Define the base indices over which to sum for the first feature,
        # based on r0 (e.g. NAN)
        basel = [0]
        for rk in r0:
            basel2 = []
            for j in range(4):
                basel2 += list(map(lambda x: x+j*rk, basel))
            basel = basel2.copy()

        # Loop over the shifts in the previous baselien induced by r1
        # (e.g. NAN, NCN, NGN, NTN) and build the features
        # Be careful with the order of features
        shiftl = [0]
        for rk in r1:
            shiftl2 = []
            for j in range(4):
                shiftl2 += list(map(lambda x: x+j*rk, shiftl))
            shiftl = shiftl2.copy()

        # print('Base', basel)
        # print('Shifts', shiftl)

        for sh in shiftl:
            to_sum_over = list(map(lambda x: x+sh, basel))
            # print('to_sum_over', to_sum_over)
            # print(sf[to_sum_over])
            phi.append(np.sum(sf[to_sum_over]))

        return phi

    def generate_lists(self):
        l = self.l
        e = '{0:0'+str(l-1)+'b}'
        li = list(range(2**(l-1)))
        li = list(map(lambda x: e.format(x), li))
        li = list(map(lambda x: x+'1', li))
        li.reverse()
        return li


    def split_list_0_1(self, li):
        r0, r1 = [], []
        for i in range(len(li)):
            if (li[i]=='0'):
                r0.append(i)
            if (li[i]=='1'):
                r1.append(i)
        return r0,r1