import numpy as np
import time
import pandas as pd
from RBT_prove import RBTree
import preprocessing
import preprocessing_noise
import os.path
import pandas as pd
from math import *
import openpyxl
import seaborn as sns
import matplotlib.pylab as plt
import pickle

# COSE DA FARE: cross validation per cercare milgiore numero di stimatore in base all'accuracy e al numero di feature usate complessivamente e anche ad ogni e livello

class Measures():
    def __init__(self, n_trees, depth, exp_folder, folder_ds, n, level_flag, threshold, pr=1):
        self.n_trees = n_trees
        self.depth = depth
        self.exp_folder = exp_folder
        self.folder_ds = folder_ds
        self.n = n
        self.level_flag = level_flag
        self.threshold = threshold
        self.pr = pr

        if self.depth == 2:
            self.dict_nodes = {0: 0, 1: 1, 2: 2} #self.dict_nodes = {0: 0, 1: 1, 2: 4} #CAMBIARE
        elif self.depth == 3:
            self.dict_nodes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6} #self.dict_nodes = {0: 0, 1: 1, 2: 8, 3: 2, 4: 5, 5: 9, 6: 12}
        elif self.depth == 4:
            self.dict_nodes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14}#self.dict_nodes = {0: 0, 1: 1, 2: 16, 3: 2, 4: 9, 5: 17, 6: 24, 7: 3, 8: 6, 9: 10, 10: 13, 11: 18, 12: 21, 13: 25, 14: 28, 15: 4}
        elif self.depth == 5:
            self.dict_nodes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                               13: 13, 14: 14, 15: 15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,28:28,29:29,30:30}

        T = (2 ** (depth + 1) - 1)
        self.n_branch = (floor(T / 2))  # 2**(d-1)

        self.Tb_level = [list(np.arange(2 ** i - 1, 2 ** (i + 1) - 1)) for i in range(self.depth)]

    def frequencies(self,weighted=False):

        df_nodes = pd.read_csv(self.exp_folder + '/' + self.folder_ds + 'df_nodes.csv')
        df_nodes = df_nodes.values.T
        times_weighted = pd.read_csv(self.exp_folder + '/' + self.folder_ds + 'weights_times.csv') # index [(feature,depth)] gives the number of times weighted

        if not self.level_flag: # PER NODO

            f = [{j: df_nodes[self.dict_nodes[t]][j] / self.n_trees for j in range(self.n) if df_nodes[self.dict_nodes[t]][j] > 0} for t in range(self.n_branch)]  # mi salvo frequenza di ogni feature per nodo

            F_base = [[j for j in f[t]] for t in range(self.n_branch)]
            F_base = [list(set(elem)) for elem in F_base]

            if self.threshold == 0.0:
                F = F_base
            else:
                perc = [np.percentile(list(f[t].values()), self.threshold) for t in range(self.n_branch)]
                F = [[j for j in f[t] if f[t][j] >= perc[t]] for t in range(self.n_branch)]
                F = [list(set(elem)) for elem in F]

        else:   # PER LIVELLO
            if weighted:
                f_level = [{} for l in range(self.depth)]  # mi salvo frequenza di ogni feature per livello
                for l in range(self.depth):
                        for j in range(self.n):  # per ogni feature
                            if j in f_level[l]:
                                if len(times_weighted[(times_weighted['Feature']==str(j))&(times_weighted['depth']==l)]['Freq'].values)>0:
                                    f_level[l][j] += times_weighted[(times_weighted['Feature']==str(j))&(times_weighted['depth']==l)]['Freq'].values[0]  # n_estimators
                            else:
                                if len(times_weighted[(times_weighted['Feature'] == str(j)) & (times_weighted['depth'] == l)]['Freq'].values)>0:
                                    f_level[l][j] = times_weighted[(times_weighted['Feature']==str(j))&(times_weighted['depth']==l)]['Freq'].values[0]  # n_estimators
                self.f_level_0 = [{j: f_level[l][j] / (2 ** l) for j in f_level[l]} for l in
                                  range(self.depth)]
            else:
                f_level = [{} for l in range(self.depth)]  # mi salvo frequenza di ogni feature per livello
                for l in range(self.depth):
                    for t in self.Tb_level[l]: # PER OGNI NODO DELLA DEPTH l
                        for j in range(self.n): # per ogni feature
                            if j in f_level[l]:
                                f_level[l][j] += df_nodes[self.dict_nodes[t]][j]  # n_estimators
                            else:
                                f_level[l][j] = df_nodes[self.dict_nodes[t]][j]  # n_estimators
                self.f_level_0 = [{j: f_level[l][j] / (self.n_trees * 2 ** l) for j in f_level[l]} for l in
                                  range(self.depth)]
            
            
            f_level = [{j: self.f_level_0[l][j] for j in self.f_level_0[l] if self.f_level_0[l][j] > 0} for l in range(self.depth)]

            F_base = [[j for j in f_level[l] if f_level[l][j] > 0] for l in range(self.depth)]
            F_base = [list(set(elem)) for elem in F_base]

            if self.threshold == 0.0:
                F = F_base
            else:
                perc_lev = [np.percentile(list(f_level[l].values()), self.threshold) for l in range(self.depth)]
                F = [[j for j in f_level[l] if f_level[l][j] >= perc_lev[l]] for l in range(self.depth)]
                F = [list(set(elem)) for elem in F]

        F_card = [len(elem) for elem in F]

        if self.level_flag:
            f = f_level

        return f, F, F_card, F_base, self.f_level_0

    def minimal_depth(self):

        df_nodes_branched = pd.read_csv(self.exp_folder + self.folder_ds + 'df_nodes_branched.csv')
        df_nodes_branched = df_nodes_branched.values[0]

        level_branched = [0 for _ in range(self.depth)]
        for level in range(self.depth):
            for t in self.Tb_level[level]:
                level_branched[level] += df_nodes_branched[self.dict_nodes[t]]

        mind_level = [[0 for j in range(self.n)] for level in range(self.depth)]

        for level in range(self.depth):
            for j in range(self.n):
                if level > 0:
                    mind_level[level][j] = prod(
                        (1 - self.pr * self.f_level_0[lev][j]) ** level_branched[lev] for lev in
                        range(level)) * (1 - (1 - self.pr * self.f_level_0[level][j]) ** level_branched[level])
                else:
                    mind_level[0][j] = (1 - (1 - self.pr * self.f_level_0[0][j]) ** level_branched[0])

        mind_level = [{j: mind_level[l][j] for j in range(len(mind_level[l])) if mind_level[l][j] > 1e-6} for l in range(self.depth)]

        F_base = [[j for j in mind_level[level] if mind_level[level][j] > 0] for level in range(self.depth)]
        F_base = [list(set(elem)) for elem in F_base]

        if self.threshold == 0.0:
            F = F_base
        else: #TODO!! TOGLIERE 0 ?
            perc_lev = [np.percentile(list(mind_level[l].values()), self.threshold) for l in range(self.depth)]
            F = [[j for j in mind_level[l] if mind_level[l][j] >= perc_lev[l]] for l in range(self.depth)]
            F = [list(set(elem)) for elem in F]

        F_card = [len(elem) for elem in F]

        return mind_level, F, F_card, F_base

    def predict_prob(self):

        p = pd.read_csv(self.exp_folder + self.folder_ds + 'y_train_pred_proba.csv', header=0, index_col=0)  # le roundiamo? todo
        p = p.values.T[0]

        return p

    def perf_RF(self):
        df_rf = pd.read_csv(self.exp_folder + self.folder_ds + 'stat_RF.csv', header=0)
        acc_train_RF = float(df_rf['train_acc_RF'].iloc[0])
        acc_test_RF = float(df_rf['test_acc_RF'].iloc[0])
        cm_train_RF = df_rf['train_cm_RF'].iloc[0]
        cm_test_RF = df_rf['test_cm_RF'].iloc[0]

        return acc_train_RF, acc_test_RF, cm_train_RF, cm_test_RF

    def proximity_measures_load(self,x,x_test,threshold):
        #m_rf = pd.read_csv('./experiments/Retino_estimators200_depth3/proxMat_noWeighted_train.csv', header=0,
        #                   index_col=0)  # le roundiamo? todo
        #m_rf_test = pd.read_csv('./experiments/Retino_estimators200_depth3/proxMat_noWeighted_test.csv', header=0,
        #                        index_col=0)  # le roundiamo? todo

        #m_rf = m_rf.values
        #m_rf_test = m_rf_test.values
        #m_rf = 0
        m_rf_test = 0
        U_rf_test = []
        Un_rf_test = []

        with open('./experiments/Retino_estimators200_depth3/U_rf'+str(threshold)+'.pkl', 'rb') as file:
            U_rf = pickle.load(file)
        with open('./experiments/Retino_estimators200_depth3/Un_rf'+str(threshold)+'.pkl', 'rb') as file:
            Un_rf = pickle.load(file)
        #with open('./experiments/Retino_estimators200_depth3/U_rf_test'+str(threshold)+'.pkl', 'rb') as file:
        #    U_rf_test = pickle.load(file)
        #with open('./experiments/Retino_estimators200_depth3/Un_rf_test'+str(threshold)+'.pkl', 'rb') as file:
        #    Un_rf_test = pickle.load(file)
        #with open('./experiments/Retino_estimators200_depth3/m_rf_test'+str(threshold)+'.pkl', 'rb') as file:
         #   m_rf_test = pickle.load(file)
        with open('./experiments/Retino_estimators200_depth3/m_rf'+str(threshold)+'.pkl', 'rb') as file:
            m_rf = pickle.load(file)


        U_rf = [tuple(row) for row in U_rf]
        print(type(U_rf))
        print(U_rf)
        Un_rf = [tuple(row) for row in Un_rf]
        U_rf_len = len(U_rf)
        Un_rf_len = len(Un_rf)
        U_rf_len_test = len(U_rf_test)
        Un_rf_len_test = len(Un_rf_test)

        return m_rf, m_rf_test, U_rf, Un_rf, U_rf_len, Un_rf_len, U_rf_test, Un_rf_test, U_rf_len_test, Un_rf_len_test

    def proximity_measures_mod(self,threshold):
        m_rf = pd.read_csv(self.exp_folder + self.folder_ds + 'proxMat_weighted_train.csv', header=0, index_col=0)  # le roundiamo? todo
        #m_rf_test = pd.read_csv('./experiments/Retino_estimators200_depth3/proxMat_weighted_test.csv', header=0,
        #                        index_col=0)  # le roundiamo? todo
        m_rf_test = []
        m_rf = m_rf.values
        #m_rf_test = m_rf_test.values

        U_rf = [(i, k) for i in range(len(m_rf)) for k in range(len(m_rf)) if k > i and m_rf[i, k] >= threshold]
        U_rf_len = len(U_rf)

        #U_rf_test = [(i, k) for i in range(len(m_rf_test)) for k in range(len(m_rf_test)) if
        #             k > i and m_rf_test[i, k] >= threshold]
        U_rf_test =[]
        U_rf_len_test = len(U_rf_test)

        Un_rf = [(i, k) for i in range(len(m_rf)) for k in range(len(m_rf)) if k > i and m_rf[i, k] == 0]
        Un_rf_len = len(Un_rf)

        #Un_rf_test = [(i, k) for i in range(len(m_rf_test)) for k in range(len(m_rf_test)) if
         #             k > i and m_rf_test[i, k] == 0]
        Un_rf_test = []
        Un_rf_len_test = len(Un_rf_test)

        return m_rf, m_rf_test, U_rf, Un_rf, U_rf_len, Un_rf_len, U_rf_test, Un_rf_test, U_rf_len_test, Un_rf_len_test

    def proximity_measures(self, x, x_test,threshold):

        m_rf = pd.read_csv(self.exp_folder + self.folder_ds + 'proxMat_weighted_train.csv', header=0, index_col=0)  # le roundiamo? todo
        m_rf_test = pd.read_csv(self.exp_folder + self.folder_ds + 'proxMat_weighted_test.csv', header=0, index_col=0)  # le roundiamo? todo
        m_rf = m_rf.values
        m_rf_test = m_rf_test.values

        U_rf = [(i, k) for i in range(len(x)) for k in range(len(x)) if k > i and m_rf[i, k] >= threshold]
        U_rf_len = len(U_rf)

        U_rf_test = [(i, k) for i in range(len(x_test)) for k in range(len(x_test)) if k > i and m_rf_test[i, k] >= threshold]
        U_rf_len_test = len(U_rf_test)

        Un_rf = [(i, k) for i in range(len(x)) for k in range(len(x)) if k > i and m_rf[i, k] == 0]
        Un_rf_len = len(Un_rf)

        Un_rf_test = [(i, k) for i in range(len(x_test)) for k in range(len(x_test)) if k > i and m_rf_test[i, k] == 0]
        Un_rf_len_test = len(Un_rf_test)

        return m_rf, m_rf_test, U_rf, Un_rf, U_rf_len, Un_rf_len, U_rf_test, Un_rf_test, U_rf_len_test, Un_rf_len_test,threshold

    def proximity_cluster(self, U_rf):

        clusters = [[]]
        for u in U_rf:
            present = False
            i = 0
            while not present and i < len(clusters):
                if u[0] in clusters[i] or u[1] in clusters[i]:
                    present = True
                else:
                    i += 1
                    
            if present:
                clusters[i].append(u[0])
                clusters[i].append(u[1])
            else:
                clusters += [[u[0],u[1]]]

        clusters = [list(set(elem)) for elem in clusters][1:]
        len_clust = [len(elem) for elem in clusters]
        max_clust = clusters[np.argmax(len_clust)]
            
        return max_clust

