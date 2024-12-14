import numpy as np
import time
import pandas as pd
from RBT_prove import RBTree
from measures import Measures
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

path = ''
folder = "prova/"
if not os.path.isdir(folder):
    os.mkdir(folder)
path = '' + folder

folder2 = "Grafici/"
if not os.path.isdir(path + folder2):
    os.mkdir(path + folder2)
path2 = path + folder2

date = 'Sept14' #DA MODIFICARE
datasets = ['Diabetes'] #DA MODIFICARE #['Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino',
           # 'Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino',
           # 'Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino',
           # 'Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino',
           # 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino',
           # 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino',
           # 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino',
           # 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino', 'Retino',
#'Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino',
           # 'Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino',
           # 'Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino',
           # 'Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino','Retino'
           # ]#'db_retino_ridotto']#['Cleveland','Diabetes','German','Heart','IndianLiver','Ionosphere','Parkinson','Sonar','Wholesale','Wisconsin'] ##['Wisconsin', 'Wisconsin_Prognostic', 'Wholesale','Parkinson']

level_flags = [True]
version = 'multivariate'
freq_weighted = True
exp_folder = 'experiments/'
measures = ['f'] #,'md'] #,'md']
n_trees = 200                  #DA MODIFICARE numero alberi della foresta
depths = [2]  # [2, 3, 4]              #DA MODIFICARE depth massima della foresta
comp = 'rf'
FSE = None
FSBs = [True]  # , False]  #DA MODIFICARE se voglio il budget massimo di feature da usare ad ogni nodo
f_flag = True
m_flags = [True]
B = 4                  #DA MODIFICARE  budget massimo di features
thresh_prox_meas_list = [0.85] #DA MODIFICARE #[0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,
                         #0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,
                         #0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,
                         #0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,
                         #0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83,0.83, 0.83,
                         #0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83,0.83, 0.83,
                         #0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83,0.83, 0.83,
                         #0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83,0.83, 0.83,
                         #0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,0.85, 0.85,
                         #0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,0.85, 0.85,
                         #0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,0.85, 0.85,
                         #0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,0.85, 0.85]
alphas =[0.5] #DA MODIFICARE #[0.6,0.5,0.4,0.3,0.2,0.1,0.7,0.8,0.9,1,
         #0.6,0.5,0.4,0.3,0.2,0.1,0.7,0.8,0.9,1,
         #0.6,0.5,0.4,0.3,0.2,0.1,0.7,0.8,0.9,1,
         #0.6,0.5,0.4,0.3,0.2,0.1,0.7,0.8,0.9,1,
         #0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.7, 0.8, 0.9, 1,
         #0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.7, 0.8, 0.9, 1,
         #0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.7, 0.8, 0.9, 1,
         #0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.7, 0.8, 0.9, 1,
         #0.6,0.5,0.4,0.3,0.2,0.1,0.7,0.8,0.9,1,
         #0.6,0.5,0.4,0.3,0.2,0.1,0.7,0.8,0.9,1,
         #0.6,0.5,0.4,0.3,0.2,0.1,0.7,0.8,0.9,1,
         #0.6,0.5,0.4,0.3,0.2,0.1,0.7,0.8,0.9,1
         #]#,0.2,0.2,0.5,0.2,0.2,0.5,0.6,0.2,0.2]
#[0.5,0.5,0.5,0.5,0.2,0.4,0.2,0.8,0.4,0.5] #[0.4],[0.2],[0.4],[0.2,0.2],[0.8],[0.4],[0.5]]
# depth 3 [0.5,0.2,0.8,0.2,0.2,0.5,0.5,0.5,0.2,0.5]
thresholds = [100/4] #DA MODIFICARE #[100/3,100/3,100/3,100/3,100/3,100/3,100/3,100/3,100/3,100/3,
              #100/2,100/2,100/2,100/2,100/2,100/2,100/2,100/2,100/2,100/2,
              #100/4,100/4,100/4,100/4,100/4,100/4,100/4,100/4,100/4,100/4,
              #0,0,0,0,0,0,0,0,0,0,
              #100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3,
              #100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2,
              #100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4,
              #0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              #100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3, 100 / 3,
              #100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2, 100 / 2,
              #100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4, 100 / 4,
              #0, 0, 0, 0, 0, 0, 0, 0, 0, 0
              #]#,50,50,100/3,0,100/3,25,25,25,50] #[100/3,100/3,25,100/3,50,50,100/3,0,0,100/3] #[25],[50],[50],[50,100/3],[0],[0],[50]]
# depth 3 [100/3,100/3,25,50,50,0,100/3,25,50,100/3]
# depth 2 [100/3,50,50,100/3,0,100/3,25,25,25,50]
data = np.array([])
name = 'RBT'
cuts, presolve = True, True
warm_starts = [None]
time_limit_s = 0
time_limit = 90*60 #60*8  # 30*60  # 3 #60*60N  #DA MODIFICARE in secondi
flag_clust = False
sym_global = True

date = date+'_alpha'+str(alphas[0])+['_Freq_noweighted','_Freq_weighted'][int(freq_weighted)]+'budget_'+str(FSBs)+'_'+str(B)+'_depth_'+str(depths)

# synthetic datasets:
size = 500  # 8 #200 # 100
margin = [0.05, 0.14,
          0.06]  # None #0.075 #[0.075, 0.075, 0.06] #[0.075, 0.075, 0.06] #[0.05, 0.14, 0.06] #[0.075, 0.075, 0.06] #
columns = ['Date', 'Model', 'Folder', 'Dataset', '(P,n)', '(P-1,P1)', 'benchmark', 'measure','depth', 'Level_flag', 'F base', 'F',
           '#F',
           'FSB', 'B', 'f', 'alpha', 'Threshold','Thres_prox_meas', 'U rf len', 'U rf len test', 'UN rf len', 'UN rf len test',
           'UvsU train', 'UvsU test', 'UvsU N train', 'UvsU N test', 'm comp train',
           'm comp test', 'N nodes','Obj value',
           'Train ACC risp TRUTH', 'Test ACC risp TRUTH', 'Train CM risp TRUTH', 'Test CM risp TRUTH',
           'Train ACC risp RF', 'Test ACC risp RF', 'Train CM risp RF', 'Test CM risp RF',
           'Train ACC RF', 'Test ACC RF', 'Train CM RF', 'Test CM RF',
           'Total time', 'Gap', 'N feat', 'feat_used_tot', 'l_feat_used', 'n_l_feat_used', 'Var star','f0']

try:
    df_prev = pd.read_excel(path + 'Stat_' + str(name) + '_' + str(date) + '.xlsx', index_col=False, header=True)
except:
    row = np.zeros(len(columns)).reshape(1, -1)
    df_prev = pd.DataFrame(row, columns=columns)

# data = pd.read_excel('C:/Users/marta/OneDrive/Desktop/RFT_8Nov/Stat_RBT_8Nov.xlsx',header=0,index_col=0)  # le roundiamo? todo

for measure in measures:
    for c in range(len(datasets)):
        thresh_prox_meas = thresh_prox_meas_list[c]
        dataset = datasets[c]
        for depth in depths:
            if depth == 2:
                folder_ds = dataset + '_estimators200_depth2/'
            elif depth == 3:
                folder_ds = dataset + '_estimators200_depth3/'
            elif depth == 4:
                folder_ds = dataset + '_estimators638_depth4/'
            elif depth == 9:
                folder_ds = dataset + '_estimators1644_depth9/'
            for level_flag in level_flags: #TODO
                    alpha = alphas[c]
                    threshold = thresholds[c]
                    for FSB in FSBs:
                            if dataset in ['3a', '3b', '4old', '4', '4unb', '6', '4stripes']:
                                x, x_test, y, y_test = preprocessing.generate_dataset(shape=dataset, size=size,
                                                                                              margin=margin,
                                                                                              test_size=0.20)
                            else:
                                x, x_test, y, y_test = preprocessing.preprocessing_data(dataset=dataset, test_size=0.20,
                                                                                                exp_folder = exp_folder, folder=folder_ds)  # TODO!!
                                y_train_rf = pd.read_csv(exp_folder + folder_ds + 'y_train_pred.csv',
                                                             index_col=0,
                                                             header=0)
                                y_train_rf = y_train_rf.values.reshape(-1, )
                                classes = np.unique(y_train_rf)
                                y_train_rf[y_train_rf == max(classes)] = 1
                                y_train_rf[y_train_rf == min(classes)] = -1

                                y_test_rf = pd.read_csv(exp_folder + folder_ds + 'y_test_pred.csv', index_col=0,
                                                            header=0)
                                y_test_rf = y_test_rf.values.reshape(-1, )
                                classes = np.unique(y_test_rf)
                                y_test_rf[y_test_rf == max(classes)] = 1
                                y_test_rf[y_test_rf == min(classes)] = -1

                                print('\n************* DATASET %s ******************\n' % (dataset))
                                P, n = len(x)+len(x_test), len(x[0])
                                y_tot = np.hstack((y, y_test))
                                Pm1, P1 = len(y_tot[y_tot == -1]) / (len(x) + len(x_test)), len(
                                        y_tot[y_tot == 1]) / (
                                                      len(x) + len(x_test))
                                print(dataset, P, n, np.round(Pm1, 2) * 100, np.round(P1, 2) * 100)

                                rbt_tree = RBTree(depth=depth, version=version, name=name)
                                    
                                meas = Measures(n_trees, depth, exp_folder, folder_ds, n, level_flag, threshold)

                                f, F, F_card, F_base, f0 = meas.frequencies(weighted=freq_weighted)

                                p = meas.predict_prob()
                                #m_rf, m_rf_test, U_rf, Un_rf, U_rf_len, Un_rf_len, U_rf_test, Un_rf_test, U_rf_len_test, Un_rf_len_test = meas.proximity_measures(x, x_test) #da modificare
                                m_rf, m_rf_test, U_rf, Un_rf, U_rf_len, Un_rf_len, U_rf_test, Un_rf_test, U_rf_len_test, Un_rf_len_test = meas.proximity_measures_mod(thresh_prox_meas)
                                #m_rf, m_rf_test, U_rf, Un_rf, U_rf_len, Un_rf_len, U_rf_test, Un_rf_test, U_rf_len_test, Un_rf_len_test = 0,0 ,[],[],0,0,[],[],0,0  # meas.proximity_measures_load(x, x_test)  # da modificare
                                #m_rf, m_rf_test, U_rf, Un_rf, U_rf_len, Un_rf_len, U_rf_test, Un_rf_test, U_rf_len_test, Un_rf_len_test = meas.proximity_measures_load(x, x_test,thresh_prox_meas)  # da modificare

                                max_clust = None #meas.proximity_cluster(U_rf)
                                                                
                                if measure == 'md':
                                    f, F, F_card, F_base = meas.minimal_depth()

                                if comp == 'truth':
                                    rbt_tree.model(x, y, dataset, level_flag, F, p, f, alpha, FSB, FSE, B,
                                                       U_rf)  # warm_start = warm_start, time_limit_ws = time_limit_s, log_flag_ws = False)            #create the model
                                    obj_value, time_opt, time_ws, mip_gap, n_feat_used, feat_used_tot, l_feat_used, n_l_feat_used, var_dict, acc_train_ws, acc_test_ws, mip_gap_ws = rbt_tree.solve(
                                            time_limit=time_limit, cuts=cuts, presolve=presolve)
                                else:
                                    rbt_tree.model(x, y_train_rf, dataset, level_flag, F, p, f, alpha, FSB,
                                                       FSE, B,
                                                       U_rf, max_clust=max_clust, flag_clust = flag_clust, sym_global = sym_global)  # warm_start = warm_start, time_limit_ws = time_limit_s, log_flag_ws = False)            #create the model

                                   # rbt_tree.m.write(path + 'mod_'+str(measure)+'_sym_global_s_vinc_'+str(dataset)+'_d'+str(depth)+'_B'+str(FSB)+'.lp')

                                  #  mp = rbt_tree.m.presolve()
                                  #  mp.write(path + 'mod_presolve_GW2.lp')
                                   # mp.write(path + 'mod_presolve_'+str(measure)+'_sym_global_s_vinc_'+str(dataset)+'_d'+str(depth)+'_B'+str(FSB)+'.lp')

                                    n_nodes, obj_value, time_opt, time_ws, mip_gap, n_feat_used, feat_used_tot, l_feat_used, n_l_feat_used, var_dict, acc_train_ws, acc_test_ws, mip_gap_ws = rbt_tree.solve(
                                           time_limit=time_limit, cuts=cuts, presolve=presolve)
                                print(n_nodes)
                                print('qui')

                                  #  rbt_tree.m.write(path + 'mod_cutted_'+str(measure)+'_sym_global_s_vinc_'+str(dataset)+'_d'+str(depth)+'_B'+str(FSB)+'.lp')
                                if obj_value is None: continue  # todo??
                                a, b, z, s = var_dict['a'], var_dict['b'], var_dict['z'], var_dict['s']
                                print('vars_dict')
                                #m_tree, U_tree, Un_tree = rbt_tree.proximity(x)
                                #m_tree_test, U_tree_test, Un_tree_test = rbt_tree.proximity(x_test)
                                m_tree, U_tree, Un_tree = rbt_tree.proximity(x)
                                #m_tree_test, U_tree_test, Un_tree_test = rbt_tree.proximity(x_test) #BOH

                                print('prossimità')
                                #print('U_rf ',U_rf)
                                #print('U_tree ',U_tree)

                                pd.DataFrame(m_tree).to_csv(exp_folder + folder_ds + "m_tree_L" + str(level_flag) + '_alpha' + str(alpha) + ".csv")
                                #pd.DataFrame(m_tree_test).to_csv(exp_folder + folder_ds + 'm_tree_test_L' + str(level_flag) + '_alpha' + str(alpha) + ".csv")

                                mcomp_train, Ucomp_train, Ucomp_n_train = rbt_tree.compare_proximity(thresh_prox_meas, x, U_tree, Un_tree, m_tree, U_rf, Un_rf, m_rf)  #da modificare
                                #mcomp_test, Ucomp_test, Ucomp_n_test = rbt_tree.compare_proximity(x_test, U_tree_test, Un_tree_test, m_tree_test, U_rf_test, Un_rf_test, m_rf_test) #da modificare
                                #mcomp_train, Ucomp_train, Ucomp_n_train = [],[],[]
                                mcomp_test, Ucomp_test, Ucomp_n_test = [],[],[]

                                acc_train_RF, acc_test_RF, cm_train_RF, cm_test_RF = meas.perf_RF()
                                print('acc')
                                pred_train, pred_test = rbt_tree.prediction(x), rbt_tree.prediction(x_test)
                                acc_train, cm_train, prec_train, recall_train, TNR_train, TPR_train, AUC_train = rbt_tree.performances(
                                    y, pred_train)
                                acc_test, cm_test, prec_test, recall_test, TNR_test, TPR_test, AUC_test = rbt_tree.performances(
                                    y_test, pred_test)

                                acc_train_vs_rf, cm_train_vs_rf, prec_train_vs_rf, recall_train_vs_rf, TNR_train_vs_rf, TPR_train_vs_rf, AUC_train_vs_rf = rbt_tree.performances(
                                    y_train_rf, pred_train)
                                acc_test_vs_rf, cm_test_vs_rf, prec_test_vs_rf, recall_test_vs_rf, TNR_test_vs_rf, TPR_test_vs_rf, AUC_test_vs_rf = rbt_tree.performances(
                                    y_test_rf, pred_test)
                                print('accs')
                                if n == 2:  # only training set ## da modificare
                                   if depth <= 3:
                                       rbt_tree.draw_plot(x, y, size, margin, type='train', date=date, path=path)
                                   elif depth == 4:
                                       rbt_tree.draw_plot4(x, y, size, margin, type='train', date=date, path=path)

                                rbt_tree.draw_graph(x, y, phase='train_b_' + comp + '_rispTRUTH_t' + str(threshold),
                                                   date=date, path=path2)  # verificare
                                rbt_tree.draw_graph(x_test, y_test,
                                                   phase='test_b_' + comp + '_rispTRUTH_t' + str(threshold),
                                                   date=date, path=path2)  # verificare

                                rbt_tree.draw_graph(x, y_train_rf,
                                                   phase='train_b_' + comp + '_rispRF_t' + str(threshold),
                                                   date=date, path=path2)  # verificare
                                rbt_tree.draw_graph(x_test, y_test_rf,
                                                   phase='test_b_' + comp + '_rispRF_t' + str(threshold),
                                                   date=date, path=path2)  # verificare

                                with open('var_dict'+str(date)+'.pickle', 'wb') as handle:
                                    pickle.dump(var_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                                row = np.array(
                                    [date, name, folder, dataset, (P, n), (Pm1, P1), comp, measure, depth, level_flag,
                                     F_base, F,
                                     F_card, FSB, B, f, alpha, threshold,thresh_prox_meas, U_rf_len,
                                     U_rf_len_test, Un_rf_len, Un_rf_len_test, Ucomp_train,
                                     Ucomp_test, Ucomp_n_train, Ucomp_n_test, mcomp_train, mcomp_test, n_nodes, obj_value,
                                     acc_train, acc_test, cm_train, cm_test,
                                     acc_train_vs_rf, acc_test_vs_rf, cm_train_vs_rf, cm_test_vs_rf, acc_train_RF,
                                     acc_test_RF, cm_train_RF, cm_test_RF, time_opt, mip_gap, n_feat_used,
                                     feat_used_tot,
                                     l_feat_used, n_l_feat_used, var_dict, f0], dtype=object)


                                row = row.reshape(1, -1)

                                df = pd.DataFrame(row, columns=columns)
                                df_tot = pd.concat([df_prev, df], ignore_index=True)
                                df_prev = df_tot
                                df_tot.to_excel(path + 'Stat_' + str(name) + '_' + str(date) + '.xlsx', index=False,
                                                header=True)

                                print('TOTAL TIME OPT: ' + str(time_opt))
                                print('ACC TRAIN', acc_train)
                                print('ACC TEST', acc_test)

