import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import math
import random
from sklearn.utils import shuffle

# Heart-disease-Cleveland         FATTO
# Hepatitis
# dermatology
# fertility                      FATTO
# Breast-cancer-diagnostic 569 30 2 90.5 94.0 +3.50 ± 0.59       FATTO
# Breast-cancer-prognostic 194 32 2 75.5 75.5 0.00 ± 0.00
# Breast-cancer 683
# Thoracic-surgery                       #FATTO

# CAD e scoliosi

def preprocessing_data(dataset, test_size=.25, flag=None, exp_folder = '', folder = ''):
    if dataset == 'scoliosi':
        df = pd.read_excel('datasets/' + dataset + '.xlsx', header=0, index_col=0)
    elif dataset in ['bands','breast_cancer_diagnostic', 'breast_cancer_prognostic', 'breast_cancer_wisconsin',
                     'breast_cancer_wisconsin_orig']:
        df = pd.read_csv('datasets/' + dataset + '.csv', header=None, index_col=0)
    elif dataset in ['parkinsons.data', 'spambase.data','ozone1','ozone2']:
        df = pd.read_csv('datasets/' + dataset + '.csv', header=0, index_col=0)
    elif dataset in ['bcd_train','boston','boston_train','wholesale', 'retinopatia_500', 'retinopatia_1000', 'retinopatia_1000_subset']:
        df = pd.read_csv('datasets/' + folder + dataset + '.csv', header=0)
    elif dataset in ['monks1','monks2','monks3']:
        df = pd.read_csv('datasets/' + dataset + '.csv', header=None, index_col=7)
    elif dataset in ['Retino','db_retino_ridotto','Diabetes','Musk','Sonar','IndianLiver','Parkinson','Wisconsin','Wisconsin_Prognostic','Wholesale','Cleveland','Ionosphere','German','Heart']: #TODO: AGGIUNGERE NOME DEL DB
        df = pd.read_csv(exp_folder + folder + 'training_data.csv', header=0)
    else:
        
        df = pd.read_csv('datasets/' + dataset + '.csv', header=None)

    # if dataset in ['bcd_train','bands','credit_approval','spambase','ozone1','ozone2','german','house_votes', 'processed.cleveland', 'breast_cancer_diagnostic', 'breast_cancer_prognostic',
    #                'breast_cancer_wisconsin', 'breast_cancer_wisconsin_orig',
    #                'mammographic_masses']:  # ,'spambase']: #todo?
    #     df = df[df.apply(lambda col: col != str('?'))]
    # df.dropna(axis=0, inplace=True)  # remove NaN
    # df.drop_duplicates(inplace=True)

    if dataset in ['Retino','db_retino_ridotto','Diabetes','Musk','Sonar','IndianLiver','Parkinson','Wisconsin','Wisconsin_Prognostic','Wholesale','German','Ionosphere','Heart','Cleveland','bcd_train','boston','boston_train','bands','credit_approval','spambase','ozone1','ozone2','german','retinopatia_500', 'retinopatia_1000', 'retinopatia_1000_subset', 'australian', 'planning_relax',
                   'tic_tac_toe', 'tic_tac_toe18', 'Dataset_78features_39osservazioni',
                   'Dataset_976features_39osservazioni', 'Dataset_26features_39osservazioni', 'processed.cleveland',
                   'sonar', 'ionosphere_orig', 'data_banknote_authentication', 'bupa',
                   'scoliosi', 'breast_cancer_wisconsin', 'breast_cancer_wisconsin_orig', 'mammographic_masses',
                   'indian_liver', 'biodegradation', 'spambase.data', 'seismic_bumps', 'german']: #TODO: AGGIUNGERE NOME DEL DB
        y = df[df.columns[-1]]  # ultima colonna
        x = df[df.columns[:-1]]
    elif dataset == 'parkinsons.data':
        y = df['status']
        x = df.drop('status', axis=1)
    elif dataset == ['monks1','monks2','monks3']:
        y = df[df.columns[0]]
        x = df[df.columns[1:]]
       # x = df[df.columns[1:7]]
    else:
        y = df[df.columns[0]]  # prima colonna
        x = df[df.columns[1:]]

    # xv = x.values
    # columns_todrop = []
    # for i in range(xv.shape[1]):
    #     if len(np.unique(xv[:, i])) == 1:
    #         print('i', i)
    #         columns_todrop.append(x.columns[i])
    # x = x.drop(columns_todrop, axis=1, inplace=False)  # remove columns(features) with one single value

    x = x.values
    y = y.values
    y_new = y.copy()
    classes, counts = np.unique(y, return_counts=True)

    if len(classes) > 2:
        if dataset == 'processed.cleveland':
            y_new[y == 0] = -1
            y_new[y != 0] = 1
            y_new = y_new.astype(int)
    else:
        if dataset == 'breast_cancer_diagnostic':
            y_new[y == 'B'] = -1
            y_new[y == 'M'] = 1
            y_new = y_new.astype(int)
        elif dataset == 'breast_cancer_prognostic':
            y_new[y == 'N'] = -1
            y_new[y == 'R'] = 1  # ??
            y_new = y_new.astype(int)
        elif dataset == 'sonar':
            y_new[y == 'M'] = -1
            y_new[y == 'R'] = 1
            y_new = y_new.astype(int)
        elif dataset == 'ionosphere_orig':
            y_new[y == 'b'] = -1
            y_new[y == 'g'] = 1
            y_new = y_new.astype(int)
        elif dataset == 'biodegradation':
            print(y_new)
            y_new[y == 'NRB'] = -1
            y_new[y == 'RB'] = 1
            y_new = y_new.astype(int)
        elif dataset == 'mushrooms':
            print(y_new)
            y_new[y == 'e'] = -1
            y_new[y == 'p'] = 1
            y_new = y_new.astype(int)
        elif dataset == 'credit_approval':
            y_new[y == '-'] = -1
            y_new[y == '+'] = 1
            y_new = y_new.astype(int)
        elif dataset == 'bands':
            y_new[y == 'noband'] = -1
            y_new[y == 'band'] = 1
            y_new = y_new.astype(int)
        else:
            y_new[y == min(classes)] = -1  # spambase: 0 (not spam), seismic_bumps, german #todo cambiare? 'seismic_bumps'
            y_new[y == max(classes)] = 1

    if dataset in ['bands','credit_approval']:
        enc = OrdinalEncoder()
        enc.fit(x)
        x = enc.transform(x) #.toarray()

    if flag == 'total':
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        # x = scaler.transform(x)

        np.random.seed(1)
        x, y_new = shuffle(x, y_new, random_state=1)
        return x, x, y_new, y_new

    if dataset == 'dataset_pirata':
        x_train = x
        y_train = y_new
        x_test = x
        y_test = y_new
    elif dataset in ['spectf_train', 'spect_train']:
        x_train = x
        y_train = y_new
        if dataset == 'spectf_train':
            df = pd.read_csv('datasets/spectf_test.csv', header=None)
            y_test = df[df.columns[0]]  # prima colonna
            x_test = df[df.columns[1:]]
        elif dataset == 'spect_train':
            df = pd.read_csv('datasets/spect_test.csv', header=None)
            y_test = df[df.columns[0]]  # prima colonna
            x_test = df[df.columns[1:]]

        y_test = y_test.values
        x_test = x_test.values
        y_test[y_test == min(classes)] = -1
        y_test[y_test == max(classes)] = 1

    elif dataset in ['Retino','db_retino_ridotto','Diabetes','Musk','Sonar','IndianLiver','Parkinson','Wisconsin','Wisconsin_Prognostic','Wholesale','boston_train','bcd_train','German','Ionosphere','Cleveland','Heart']: #TODO: AGGIUNGERE NOME DEL DB #da modificare
            x_train = x
            y_train = y_new

            if dataset == 'boston_train':
                df = pd.read_csv('datasets/' + folder + 'boston_test.csv', header=0)
            elif dataset == 'bcd_train':
                df = pd.read_csv('datasets/' + folder + 'bcd_test.csv', header=0)
            else:
                df = pd.read_csv(exp_folder + folder + 'test_data.csv', header=0)
            y_test = df[df.columns[-1]]  #ultima colonna
            x_test = df[df.columns[:-1]]

            y_test = y_test.values
            x_test = x_test.values
            y_test[y_test == min(classes)] = -1
            y_test[y_test == max(classes)] = 1

    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size=test_size, shuffle=True, random_state=10,
                                                            stratify=y)
    #TODO: rimettere random_state a 0!!!!!!!!!!!!!!

    # # Scalare i dati
    # if dataset in ['boston_train','bcd_train','Heart']:
    #     print('Non scalare')
    # else:
    #     scaler = MinMaxScaler()
    #     x_train = scaler.fit_transform(x_train)
    #     x_test = scaler.transform(x_test)  # TODO ?

    return x_train, x_test, y_train, y_test


def gen_6(shape, size, margin=None, test_size=.25):
    k = int(size / 6)

    np.random.seed(1000)

    x1 = np.random.random_sample(1000).reshape(-1, 1)
    x2 = np.random.random_sample(1000).reshape(-1, 1)

    indexes_no = [i for i in range(1000) if (x1[i] - margin <= x2[i] <= x1[i] + margin) or (
            -x1[i] + 1 - margin <= x2[i] <= -x1[i] + 1 + margin) or (0.5 - margin <= x1[i] <= 0.5 + margin)]

    indexes = [i for i in range(1000) if i not in indexes_no]
    x1 = x1[indexes]
    x2 = x2[indexes]

    indeces_class = [[] for i in range(6)]

    for i in range(len(x1)):
        if x1[i] >= 0.5 and x2[i] >= x1[i]:
            indeces_class[0] += [i]
        elif x1[i] >= 0.5 and x2[i] <= -x1[i] + 1:
            indeces_class[2] += [i]
        elif x1[i] < 0.5 and x2[i] >= x1[i] and x2[i] <= -x1[i] + 1:
            indeces_class[4] += [i]
        elif x1[i] < 0.5 and x2[i] < x1[i]:
            indeces_class[1] += [i]
        elif x1[i] < 0.5 and x2[i] > -x1[i] + 1:
            indeces_class[3] += [i]
        elif x1[i] >= 0.5 and x2[i] < x1[i] and x2[i] > -x1[i] + 1:
            indeces_class[5] += [i]

    index_tot = []
    for i in range(6):
        indeces_ok = indeces_class[i][:k]
        index_tot += indeces_ok

    x2 = np.array(x2)[index_tot]
    x1 = np.array(x1)[index_tot]

    y = [1 if (x1[i] >= 0.5 and (x2[i] >= x1[i] or (x2[i] <= -x1[i] + 1))) or
              (x1[i] < 0.5 and x2[i] >= x1[i] and x2[i] <= -x1[i] + 1) else -1 for i in range(len(x1))]

    x = np.hstack((x1, x2))
    y = np.array(y)
    x_new = x
    y_new = y
    #  x_new = np.delete(x, [11, 23, 28, 39], axis = 0)
    #  y_new = np.delete(y, [11, 23, 28, 39], axis = 0)

    # print(x_new)
    return x_new, x_new, y_new, y_new


def gen_4_unbalanced(size, margin=None, test_size=.25, t=0.0):
    k = [int(size / 4), int(size / 4), int(size / 4), int(size / 4)]

    np.random.seed(1)  # 100

    x1 = np.random.random_sample(10000).reshape(-1, 1)
    x2 = np.random.random_sample(10000).reshape(-1, 1)

    m1, m2, m3 = margin[0], margin[1], margin[2]
    indexes_no = [i for i in range(10000) if (x1[i] - t - m1 <= x2[i] <= x1[i] - t + m1 and x1[i] <= 0.5) or (
                x1[i] - t - m2 <= x2[i] <= x1[i] - t + m2 and x1[i] > 0.5) or (
                          -x1[i] + 1 - m3 <= x2[i] <= -x1[i] + 1 + m3)]

    #  print(len(indexes_no))

    indexes = [i for i in range(10000) if i not in indexes_no]
    x1 = x1[indexes]
    x2 = x2[indexes]

    indeces_class = [[] for i in range(4)]

    for i in range(len(x1)):
        if x2[i] >= x1[i] - t and x2[i] >= -x1[i] + 1:
            indeces_class[0] += [i]
        elif x2[i] >= x1[i] - t and x2[i] <= -x1[i] + 1:
            indeces_class[2] += [i]
        elif x2[i] <= x1[i] - t and x2[i] >= -x1[i] + 1:
            indeces_class[3] += [i]
        elif x2[i] <= x1[i] - t and x2[i] <= -x1[i] + 1:
            indeces_class[1] += [i]

    index_tot = []
    for i in range(4):
        indeces_ok = indeces_class[i][:k[i]]
        index_tot += indeces_ok

    x2 = np.array(x2)[index_tot]
    x1 = np.array(x1)[index_tot]

    # y = [1 if (x1[i] >= 0.5 and (x2[i] >= x1[i] or (x2[i] <= -x1[i] + 1))) or
    #           (x1[i] < 0.5 and x2[i] >= x1[i] and x2[i] <= -x1[i] + 1) else -1 for i in range(len(x1))]

    y = [1 if (x2[i] >= x1[i] - t and x2[i] >= -x1[i] + 1) or (x2[i] <= x1[i] - t and x2[i] <= -x1[i] + 1) else -1
         for i in
         range(len(x1))]

    x = np.hstack((x1, x2))
    y = np.array(y)

    return x, x, y, y


def gen_6_unbalanced(size, margin=None, test_size=.25, t=0.0):
    k = [int(size / 6), int(size / 6), int(size / 6), int(size / 6), int(size / 6), int(size / 6)]

    np.random.seed(1)  # 100

    x1 = np.random.random_sample(10000).reshape(-1, 1)
    x2 = np.random.random_sample(10000).reshape(-1, 1)

    m1, m2, m3 = margin[0], margin[1], margin[2]

    indexes_no = [i for i in range(1000) if (x1[i] - m1 <= x2[i] <= x1[i] + m1) or (
            -x1[i] + 1 - m2 <= x2[i] <= -x1[i] + 1 + m2) or (0.5 - m3 <= x1[i] <= 0.5 + m3)]

    #  print(len(indexes_no))

    indexes = [i for i in range(10000) if i not in indexes_no]
    x1 = x1[indexes]
    x2 = x2[indexes]

    indeces_class = [[] for i in range(6)]

    for i in range(len(x1)):
        if x1[i] >= 0.5 and x2[i] >= x1[i]:
            indeces_class[0] += [i]
        elif x1[i] >= 0.5 and x2[i] <= -x1[i] + 1:
            indeces_class[2] += [i]
        elif x1[i] < 0.5 and x2[i] >= x1[i] and x2[i] <= -x1[i] + 1:
            indeces_class[4] += [i]
        elif x1[i] < 0.5 and x2[i] < x1[i]:
            indeces_class[1] += [i]
        elif x1[i] < 0.5 and x2[i] > -x1[i] + 1:
            indeces_class[3] += [i]
        elif x1[i] >= 0.5 and x2[i] < x1[i] and x2[i] > -x1[i] + 1:
            indeces_class[5] += [i]

    index_tot = []
    for i in range(6):
        indeces_ok = indeces_class[i][:k[i]]
        index_tot += indeces_ok

    x2 = np.array(x2)[index_tot]
    x1 = np.array(x1)[index_tot]

    y = [1 if (x1[i] >= 0.5 and (x2[i] >= x1[i] or (x2[i] <= -x1[i] + 1))) or (
            x1[i] < 0.5 and x2[i] >= x1[i] and x2[i] <= -x1[i] + 1) else -1 for i in range(len(x1))]

    x = np.hstack((x1, x2))
    y = np.array(y)

    return x, x, y, y


def gen_4v(size, margin=None, test_size=.25, t=0.0):
    k = [int(size / 4), int(size / 4), int(size / 4),
         int(size / 4)]  # [int(size / 3), int(size / 6), int(size / 3), int(size / 9)]

    np.random.seed(100)

    x1 = np.random.random_sample(10000).reshape(-1, 1)
    x2 = np.random.random_sample(10000).reshape(-1, 1)

    indexes_no = [i for i in range(10000) if (0.5 - margin <= x2[i] <= 0.5 + margin) or (
            0.5 - margin <= x1[i] <= 0.5 + margin)]

    print(len(indexes_no))

    indexes = [i for i in range(10000) if i not in indexes_no]
    x1 = x1[indexes]
    x2 = x2[indexes]

    indeces_class = [[] for i in range(4)]

    for i in range(len(x1)):
        if x2[i] >= 0.5 and x1[i] >= 0.5:
            indeces_class[0] += [i]
        elif x2[i] >= 0.5 and x1[i] < 0.5:
            indeces_class[1] += [i]
        elif x2[i] < 0.5 and x1[i] >= 0.5:
            indeces_class[2] += [i]
        elif x2[i] < 0.5 and x1[i] < 0.5:
            indeces_class[3] += [i]

    index_tot = []
    for i in range(4):
        indeces_ok = indeces_class[i][:k[i]]
        index_tot += indeces_ok

    x2 = np.array(x2)[index_tot]
    x1 = np.array(x1)[index_tot]

    y = [1 if (x2[i] >= 0.5 - t and x1[i] >= 0.5) or (x2[i] <= 0.5 and x1[i] <= 0.5) else -1
         for i in
         range(len(x1))]

    x = np.hstack((x1, x2))
    y = np.array(y)

    return x, x, y, y


def gen_4(size, margin=None, test_size=.25, t=0.0):
    k = [int(size / 4), int(size / 4), int(size / 4),
         int(size / 4)]  # [int(size / 3), int(size / 6), int(size / 3), int(size / 9)]

    np.random.seed(100)

    x1 = np.random.random_sample(10000).reshape(-1, 1)
    x2 = np.random.random_sample(10000).reshape(-1, 1)

    indexes_no = [i for i in range(10000) if (x1[i] - t - margin <= x2[i] <= x1[i] - t + margin) or (
            -x1[i] + 1 - margin <= x2[i] <= -x1[i] + 1 + margin)]

    print(len(indexes_no))

    indexes = [i for i in range(10000) if i not in indexes_no]
    x1 = x1[indexes]
    x2 = x2[indexes]

    indeces_class = [[] for i in range(4)]

    for i in range(len(x1)):
        if x2[i] >= x1[i] - t and x2[i] >= -x1[i] + 1:
            indeces_class[0] += [i]
        elif x2[i] >= x1[i] - t and x2[i] <= -x1[i] + 1:
            indeces_class[2] += [i]
        elif x2[i] <= x1[i] - t and x2[i] >= -x1[i] + 1:
            indeces_class[3] += [i]
        elif x2[i] <= x1[i] - t and x2[i] <= -x1[i] + 1:
            indeces_class[1] += [i]

    index_tot = []
    for i in range(4):
        indeces_ok = indeces_class[i][:k[i]]
        index_tot += indeces_ok

    x2 = np.array(x2)[index_tot]
    x1 = np.array(x1)[index_tot]

    y = [1 if (x2[i] >= x1[i] - t and x2[i] >= -x1[i] + 1) or (x2[i] <= x1[i] - t and x2[i] <= -x1[i] + 1) else -1
         for i in
         range(len(x1))]

    x = np.hstack((x1, x2))
    y = np.array(y)

    return x, x, y, y


#
# k = int(size/4)
#
#   np.random.seed(100)
#
#   x1 = np.random.random_sample(10000).reshape(-1, 1)
#   x2 = np.random.random_sample(10000).reshape(-1, 1)
#
#   indexes_no = [i for i in range(10000) if (x1[i] - t - margin <= x2[i] <= x1[i] - t + margin) or (
#           -x1[i] + 1 - margin <= x2[i] <= -x1[i] + 1 + margin)]
#
#   indexes = [i for i in range(10000) if i not in indexes_no]
#   x1 = x1[indexes]
#   x2 = x2[indexes]
#
#   indeces_class = [[] for i in range(4)]
#
#   for i in range(len(x1)):
#       if x2[i] >= x1[i] - t and x2[i] >= -x1[i] + 1: indeces_class[0] += [i]
#       elif x2[i] >= x1[i] - t and x2[i] <= -x1[i] + 1: indeces_class[2] += [i]
#       elif x2[i] <= x1[i] - t and x2[i] >= -x1[i] + 1: indeces_class[3] += [i]
#       elif x2[i] <= x1[i] - t and x2[i] <= -x1[i] + 1: indeces_class[1] += [i]
#
#   index_tot = []
#   for i in range(4):
#       indeces_ok = indeces_class[i][:k]
#       index_tot += indeces_ok
#
#   x2 = np.array(x2)[index_tot]
#   x1 = np.array(x1)[index_tot]
#
#   y = [1 if (x2[i] >= x1[i] - t and x2[i] >= -x1[i] + 1) or (x2[i] <= x1[i] - t and x2[i] <= -x1[i] + 1) else -1 for i in
#        range(len(x1))]
#
#   x = np.hstack((x1, x2))
#   y = np.array(y)
#
#   return x, x, y, y

def generate_dataset(shape, size, margin=None, test_size=.25):
    if shape == '6':
        return gen_6(shape, size, margin, test_size)

    if shape == '4':
        return gen_4(size, margin, test_size)

    if shape == '4v':
        return gen_4v(size, margin, test_size)

    if shape == '4unb':
        return gen_4_unbalanced(size, margin, test_size)

    if shape == '6unb':
        return gen_6_unbalanced(size, margin, test_size)

    np.random.seed(1000)

    x1 = np.random.random_sample(size).reshape(-1, 1)
    x2 = np.random.random_sample(size).reshape(-1, 1)
    # y = np.zeros(size)

    if margin is not None:
        if shape == '3a':
            indexes_no = [i for i in range(size) if
                          (x2[i] >= -x1[i] + 1 and x1[i] - margin <= x2[i] <= x1[i] + margin) or (
                                  x2[i] >= x1[i] and -x1[i] + 1 - margin <= x2[i] <= -x1[i] + 1 + margin)]
        elif shape == '3b':
            indexes_no = [i for i in range(size) if
                          (x2[i] >= -x1[i] + 1 and x1[i] - margin <= x2[i] <= x1[i] + margin) or (
                                  x2[i] <= x1[i] and -x1[i] + 1 - margin <= x2[i] <= -x1[i] + 1 + margin)]
        elif shape == '4old':
            indexes_no = [i for i in range(size) if (x1[i] - margin <= x2[i] <= x1[i] + margin) or (
                    -x1[i] + 1 - margin <= x2[i] <= -x1[i] + 1 + margin)]

        elif shape == '4b':  # 4 partizioni
            indexes_no = [i for i in range(size) if
                          (x1[i] - margin <= x2[i] <= x1[i] + margin) or (x1[i] - margin <= 0.5 <= x1[i] + margin)]

        elif shape == '6':
            indexes_no = [i for i in range(size) if (x1[i] - margin <= x2[i] <= x1[i] + margin) or (
                    -x1[i] + 1 - margin <= x2[i] <= -x1[i] + 1 + margin) or (0.5 - margin <= x1[i] <= 0.5 + margin)]

        elif shape == '8':
            indexes_no = [i for i in range(size) if (x1[i] - margin <= x2[i] <= x1[i] + margin) or (
                    -x1[i] + 1 - margin <= x2[i] <= -x1[i] + 1 + margin) or (0.5 - margin <= x1[i] <= 0.5 + margin) or (
                                      0.5 - margin <= x2[i] <= 0.5 + margin)]

        elif shape == '4stripes':
            indexes_no = [i for i in range(size) if (x1[i] - margin <= x2[i] <= x1[i] + margin) or (
                    x1[i] - 0.5 - margin <= x2[i] <= x1[i] - 0.5 + margin) or (
                                      x1[i] + 0.5 - margin <= x2[i] <= x1[i] + 0.5 + margin)]

        indexes = [i for i in range(size) if i not in indexes_no]
        x1 = x1[indexes]
        x2 = x2[indexes]

        if shape == '4old':
            ind_1a = []
            ind_1b = []
            ind_m1a = []
            ind_m1b = []
            for i in range(len(x1)):
                if (x2[i] >= x1[i] and x2[i] >= -x1[i] + 1):
                    ind_1a += [i]
                elif (x2[i] <= x1[i] and x2[i] <= -x1[i] + 1):
                    ind_1b += [i]
                elif (x2[i] >= x1[i] and x2[i] <= -x1[i] + 1):
                    ind_m1a += [i]
                elif (x2[i] <= x1[i] and x2[i] >= -x1[i] + 1):
                    ind_m1b += [i]

            print('ind1a', ind_1a)
            print('ind1b', ind_1b)
            print('indm1a', ind_m1a)
            print('indm1b', ind_m1b)

            count_1a = len(ind_1a)
            count_1b = len(ind_1b)

            count_m1a = len(ind_m1a)
            count_m1b = len(ind_m1b)

            min_tot = min(count_1a, count_m1a, count_1b, count_m1b)
            indexes = [ind_1a, ind_m1a, ind_1b, ind_m1b]
            index_tot = []
            for i in range(len(indexes)):
                if len(indexes[i]) >= min_tot:
                    diff = len(indexes[i]) - min_tot
                    print('i:', i, indexes[i][diff:])
                    index_tot += indexes[i][diff:]
                #  print('b',len(indexes[i]))

            print(len(x1))
            print(len(x2))
            # index_tot = ind_1a+ind_m1a+ind_1b+ind_m1b
            print(len(index_tot))
            index_tot.sort()
            x2 = np.array(x2)[index_tot]
            x1 = np.array(x1)[index_tot]
            print(len(x1))
            print(len(x2))
    if shape == '3a':  # 3 partizioni coerenti con nostra divisione
        y = [-1 if (x2[i] <= x1[i] or x2[i] <= -x1[i] + 1) else 1 for i in range(len(x1))]

    elif shape == '3b':  # 3 partizioni non coerenti
        y = [-1 if (x2[i] >= x1[i] or x2[i] <= -x1[i] + 1) else 1 for i in range(len(x1))]

    elif shape == '4old':  # 4 partizioni
        y = [1 if (x2[i] >= x1[i] and x2[i] >= -x1[i] + 1) or (x2[i] <= x1[i] and x2[i] <= -x1[i] + 1) else -1 for i in
             range(len(x1))]

    elif shape == '4b':  # 4 partizioni
        y = [1 if (x2[i] >= x1[i] and x1[i] >= 0.5) or (x1[i] <= 0.5 and x2[i] <= x1[i]) else -1 for i in
             range(len(x1))]

    elif shape == '6':
        y = [1 if (x1[i] >= 0.5 and (x2[i] >= x1[i] or (x2[i] <= -x1[i] + 1))) or (
                x1[i] < 0.5 and x2[i] >= x1[i] and x2[i] <= -x1[i] + 1) else -1 for i in range(len(x1))]

    elif shape == '8':
        y = [1 if (x1[i] >= 0.5 and (x2[i] >= x1[i] or (x2[i] <= 0.5 and x2[i] >= -x1[i] + 1))) or (
                x1[i] < 0.5 and (x2[i] <= x1[i] or (x2[i] >= 0.5 and x2[i] <= -x1[i] + 1))) else -1 for i in
             range(len(x1))]

    elif shape == '4stripes':
        y = [1 if ((x2[i] >= x1[i] + 0.5) or (x1[i] - 0.5 <= x2[i] <= x1[i])) else -1 for i in range(len(x1))]

    x = np.hstack((x1, x2))
    y = np.array(y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=0,
    #      stratify=y)

    return x, x, y, y


def generate_dataset_pt(shape, margin=None, test_size=.25):
    if shape == '4pt':
        l_1 = [0.25, 0.75, 0.75, 0.25]
        l_2 = [0.25, 0.25, 0.75, 0.75]

        x1 = np.array([l_1]).reshape(-1, 1)
        x2 = np.array([l_2]).reshape(-1, 1)

        x = np.hstack((x1, x2))
        y = np.array([1, -1, 1, -1])

        x_train, x_test, y_train, y_test = x, x, y, y

    if shape == '8pt':
        l_1 = [0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25, 0.25]
        l_2 = [0.25, 0.25, 0.25, 0.50, 0.75, 0.75, 0.75, 0.50]

        x1 = np.array([l_1]).reshape(-1, 1)
        x2 = np.array([l_2]).reshape(-1, 1)

        x = np.hstack((x1, x2))
        y = np.array([1, -1, 1, -1, 1, -1, 1, -1])

        x_train, x_test, y_train, y_test = x, x, y, y

    return x_train, x_test, y_train, y_test
