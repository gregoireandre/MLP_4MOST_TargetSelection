# Module that contains various preprocessing techniques for imbalanced datasets.
# The documentation for those functions can be found at : https://imbalanced-learn.readthedocs.io/en/stable/index.html

import os
import sys
import math
import numpy as np
import pandas as pd
from collections import Counter
from astropy.io import fits
from imblearn.ensemble import BalanceCascade
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler, EditedNearestNeighbours, TomekLinks, AllKNN, InstanceHardnessThreshold

def save_preprocessed_dataset(script_dir, catalog_filename, constraints, data_imputation, normalization, classification_problem, zphot_safe_threshold, train_test_val_split, cv_fold_nbr, preprocessing_methods, X_train, Y_train, data_keys_train):

    savepath = os.path.join(script_dir, 'datasets', catalog_filename, constraints + '-constraints', 'preprocessed_' + str(data_imputation) + '-imputation', classification_problem + '_' + str(zphot_safe_threshold) + '-zsafe' + '_train_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '_' + str(cv_fold_nbr) + '_norm-' + str(normalization))
    for idx, i in enumerate(preprocessing_methods):
        if idx == 0:
            filename = i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])
        else:
            filename += '_' + i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    data_keys_train += ['class']
    table_dataset_train = np.column_stack((X_train, Y_train))

    all_fits_column = []
    for g in range(table_dataset_train.shape[1]):
        all_fits_column.append(fits.Column(name=data_keys_train[g], array=table_dataset_train[:,g], format='D'))
    train = fits.BinTableHDU.from_columns(all_fits_column)

    try:
        train.writeto(os.path.join(savepath, filename + '.fits'))
    except OSError as e:
        print(e)

    return

def compute_sampling_strategy(majority_minority_ratio, Y_train, sampling_method):
    labels, counts = np.unique(Y_train, return_counts=True)
    majority_class_idx = np.argmax(counts)
    minority_class_idx = np.argmin(counts)
    majority_class_label = labels[majority_class_idx]
    sampling_strategy_dict = {}
    if sampling_method == 'undersampling':
        for idx, label in enumerate(labels):
            if label == majority_class_label:
                if (int(counts[minority_class_idx]*majority_minority_ratio) < counts[idx]):
                    sampling_strategy_dict[label] = int(counts[minority_class_idx]*majority_minority_ratio)
                else:
                    sampling_strategy_dict[label] = counts[idx]
            else:
                sampling_strategy_dict[label] = counts[idx]
    elif sampling_method == 'oversampling':
        for idx, label in enumerate(labels):
            if (label == majority_class_label):
                sampling_strategy_dict[label] = counts[idx]
            else:
                if (int(counts[majority_class_idx]/majority_minority_ratio) > counts[idx]):
                    sampling_strategy_dict[label] = int(counts[majority_class_idx]/majority_minority_ratio)
                else:
                    sampling_strategy_dict[label] = counts[idx]

    return sampling_strategy_dict

def shuffle_dataset(X, Y, seed):

    np.random.seed(seed)                                        # Fix Numpy random seed for reproducibility

    tmp_dataset = np.zeros((X.shape[0], X.shape[1] + 1))
    tmp_dataset[:,:-1] = X
    tmp_dataset[:,-1] = Y
    np.random.shuffle(tmp_dataset)
    X = tmp_dataset[:,:-1]
    Y = tmp_dataset[:,-1]

    return X, Y

def remove_imputed_datapoints(X, Y, random_seed, method_to_apply):

    dataset = np.zeros((X.shape[0], X.shape[1] + 1))
    dataset[:,:-1] = X
    dataset[:,-1] = Y

    current_module = sys.modules[__name__]
    nbr_features = dataset.shape[1] - 1
    nbr_object = dataset.shape[0]
    missing_features_dict = {}
    no_WISE = []
    sampling_strategy_noWISE = {}
    idx_noWISE = [0,1,2,3,4,5,6,8,9,10,11,12,13,14]
    no_VHS = []
    sampling_strategy_noVHS = {}
    idx_noVHS = [0,1,2,3,4,7,8,9,10,11,12,15]
    no_WISE_VHS = []
    sampling_strategy_noWISEVHS = {}
    idx_noWISEVHS = [0,1,2,3,4,8,9,10,11,12]
    complete = []


    for i in range(nbr_object):
        bool_WISE = []
        bool_VHS = []
        for j in [5,6,13,14]:
            bool_VHS.append(np.isnan(dataset[i,j]))
        bool_VHS = any(bool_VHS)
        for j in [7,15]:
            bool_WISE.append(np.isnan(dataset[i,j]))
        bool_WISE = any(bool_WISE)
        if bool_WISE and bool_VHS:
            no_WISE_VHS.append(dataset[i,idx_noWISEVHS + [-1]])
        elif bool_WISE:
            no_WISE.append(dataset[i,idx_noWISE + [-1]])
        elif bool_VHS:
            no_VHS.append(dataset[i,idx_noVHS + [-1]])
        else:
            complete.append(dataset[i,:])

    no_VHS = np.array(no_VHS)
    print(no_VHS.shape)
    no_WISE = np.array(no_WISE)
    no_WISE_VHS = np.array(no_WISE_VHS)
    complete = np.array(complete)

    labels, counts_total = np.unique(dataset[:, -1], return_counts=True)
    full_dataset_dict = dict(zip(labels, counts_total))

    labels, counts = np.unique(no_VHS[:, -1], return_counts=True)
    VHS_dict = dict(zip(labels, counts))

    labels, counts = np.unique(no_WISE[:, -1], return_counts=True)
    WISE_dict = dict(zip(labels, counts))

    labels, counts = np.unique(no_WISE_VHS[:, -1], return_counts=True)
    WISE_VHS_dict = dict(zip(labels, counts))

    labels, counts = np.unique(complete[:, -1], return_counts=True)
    complete_dict = dict(zip(labels, counts))

    object_count = 0
    print('before : ', full_dataset_dict)

    for idx, i in enumerate(method_to_apply):
        if ((not isinstance(i['arguments'][0], str)) and 'os' in i['method']):
            sampling_strategy = compute_sampling_strategy(i['arguments'][0], Y, 'oversampling')
        elif ((not isinstance(i['arguments'][0], str)) and 'us' in i['method']):
            sampling_strategy = compute_sampling_strategy(i['arguments'][0], Y, 'undersampling')

        print('sampling_strategy : ', sampling_strategy)

        nbr_object_resampled = sum(list(sampling_strategy.values()))
        X_os = np.empty((2*nbr_object_resampled,nbr_features))
        X_os[:] = np.nan
        Y_os = np.zeros(2*nbr_object_resampled)
        Y_os[:] = np.nan
        print(X_os[0,:])
        print(X_os[-1,:])

        VHS_sampling_strategy = {}
        for j in list(full_dataset_dict.keys()):
            if j in VHS_dict.keys():
                VHS_sampling_strategy[j] = math.ceil(VHS_dict[j]/full_dataset_dict[j]*sampling_strategy[j])

        print('VHS dict : ', VHS_dict)
        print('VHS sp : ', VHS_sampling_strategy)

        if len(i['arguments']) == 1:
            X_train, Y_train = getattr(current_module, i['method'])(no_VHS[:,0:-1], no_VHS[:,-1], random_seed, VHS_sampling_strategy)
        elif len(i['arguments']) == 2:
            X_train, Y_train = getattr(current_module, i['method'])(no_VHS[:,0:-1], no_VHS[:,-1], random_seed, VHS_sampling_strategy, i['arguments'][1])
        elif len(i['arguments']) == 3:
            X_train, Y_train = getattr(current_module, i['method'])(no_VHS[:,0:-1], no_VHS[:,-1], random_seed, VHS_sampling_strategy, i['arguments'][1], i['arguments'][2])
        elif len(i['arguments']) == 4:
            X_train, Y_train = getattr(current_module, i['method'])(no_VHS[:,0:-1], no_VHS[:,-1], random_seed, VHS_sampling_strategy, i['arguments'][1], i['arguments'][2], i['arguments'][3])

        X_os[object_count:object_count+X_train.shape[0], idx_noVHS] = X_train
        Y_os[object_count:object_count+X_train.shape[0]] = Y_train
        object_count += X_train.shape[0]

        WISE_sampling_strategy = {}
        for j in list(full_dataset_dict.keys()):
            if j in WISE_dict.keys():
                WISE_sampling_strategy[j] = math.ceil(WISE_dict[j]/full_dataset_dict[j]*sampling_strategy[j])

        print('WISE dict : ', WISE_dict)
        print('WISE sp : ', WISE_sampling_strategy)

        if len(i['arguments']) == 1:
            X_train, Y_train = getattr(current_module, i['method'])(no_WISE[:,0:-1], no_WISE[:,-1], random_seed, WISE_sampling_strategy)
        elif len(i['arguments']) == 2:
            X_train, Y_train = getattr(current_module, i['method'])(no_WISE[:,0:-1], no_WISE[:,-1], random_seed, WISE_sampling_strategy, i['arguments'][1])
        elif len(i['arguments']) == 3:
            X_train, Y_train = getattr(current_module, i['method'])(no_WISE[:,0:-1], no_WISE[:,-1], random_seed, WISE_sampling_strategy, i['arguments'][1], i['arguments'][2])
        elif len(i['arguments']) == 4:
            X_train, Y_train = getattr(current_module, i['method'])(no_WISE[:,0:-1], no_WISE[:,-1], random_seed, WISE_sampling_strategy, i['arguments'][1], i['arguments'][2], i['arguments'][3])

        X_os[object_count:object_count+X_train.shape[0], idx_noWISE] = X_train
        Y_os[object_count:object_count+X_train.shape[0]] = Y_train
        object_count += X_train.shape[0]

        WISE_VHS_sampling_strategy = {}
        for j in list(full_dataset_dict.keys()):
            if j in WISE_VHS_dict.keys():
                WISE_VHS_sampling_strategy[j] = math.ceil(WISE_VHS_dict[j]/full_dataset_dict[j]*sampling_strategy[j])

        print('WISE_VHS dict : ', WISE_VHS_dict)
        print('WISE_VHS sp : ', WISE_VHS_sampling_strategy)

        if len(i['arguments']) == 1:
            X_train, Y_train = getattr(current_module, i['method'])(no_WISE_VHS[:,0:-1], no_WISE_VHS[:,-1], random_seed, WISE_VHS_sampling_strategy)
        elif len(i['arguments']) == 2:
            X_train, Y_train = getattr(current_module, i['method'])(no_WISE_VHS[:,0:-1], no_WISE_VHS[:,-1], random_seed, WISE_VHS_sampling_strategy, i['arguments'][1])
        elif len(i['arguments']) == 3:
            X_train, Y_train = getattr(current_module, i['method'])(no_WISE_VHS[:,0:-1], no_WISE_VHS[:,-1], random_seed, WISE_VHS_sampling_strategy, i['arguments'][1], i['arguments'][2])
        elif len(i['arguments']) == 4:
            X_train, Y_train = getattr(current_module, i['method'])(no_WISE_VHS[:,0:-1], no_WISE_VHS[:,-1], random_seed, WISE_VHS_sampling_strategy, i['arguments'][1], i['arguments'][2], i['arguments'][3])

        X_os[object_count:object_count+X_train.shape[0], idx_noWISEVHS] = X_train
        Y_os[object_count:object_count+X_train.shape[0]] = Y_train
        object_count += X_train.shape[0]

        complete_sampling_strategy = {}
        for j in list(full_dataset_dict.keys()):
            if j in complete_dict.keys():
                complete_sampling_strategy[j] = math.ceil(complete_dict[j]/full_dataset_dict[j]*sampling_strategy[j])

        print('complete dict : ', complete_dict)
        print('complete sp : ', complete_sampling_strategy)

        if len(i['arguments']) == 1:
            X_train, Y_train = getattr(current_module, i['method'])(complete[:,0:-1], complete[:,-1], random_seed, complete_sampling_strategy)
        elif len(i['arguments']) == 2:
            X_train, Y_train = getattr(current_module, i['method'])(complete[:,0:-1], complete[:,-1], random_seed, complete_sampling_strategy, i['arguments'][1])
        elif len(i['arguments']) == 3:
            X_train, Y_train = getattr(current_module, i['method'])(complete[:,0:-1], complete[:,-1], random_seed, complete_sampling_strategy, i['arguments'][1], i['arguments'][2])
        elif len(i['arguments']) == 4:
            X_train, Y_train = getattr(current_module, i['method'])(complete[:,0:-1], complete[:,-1], random_seed, complete_sampling_strategy, i['arguments'][1], i['arguments'][2], i['arguments'][3])

        X_os[object_count:object_count+X_train.shape[0], :] = X_train
        Y_os[object_count:object_count+X_train.shape[0]] = Y_train
        object_count += X_train.shape[0]

    # labels, counts = np.unique(Y_os, return_counts=True)
    # final_dict = dict(zip(labels, counts))
    # print(final_dict)

    # idx_to_keep = [i for i in Y_os.shape[0] if not np.isnan(Y_os)]
    # id_to_keep = np.where(~np.isnan(Y_os))[0]
    # print(idx_to_delete)
    # X_os_final = X_os[:object_count, :]
    # Y_os_final = Y_os[:object_count]

    return X_os[:object_count, :], Y_os[:object_count]

def remove_extra_datapoints(X, Y, sampling_strategy):

    labels, counts = np.unique(Y, return_counts=True)
    labels_dict = dict(zip(labels, counts))

    print(labels_dict)
    print(sampling_strategy)

    for i in list(sampling_strategy.keys()):
        nbr_object_to_delete = labels_dict[i] - sampling_strategy[i]
        print(i, nbr_object_to_delete)
        if nbr_object_to_delete > 0:
            idx_label = np.where(Y==i)
            Y_reduced = Y[idx_label[nbr_object_to_delete:]]
            X_reduced = X[idx_label[nbr_object_to_delete:], :]

    return X_reduced, Y_reduced


################################################################################Oversampling Methods###############################################################################

def SMOTE_os(X_train, Y_train, seed, sampling_strategy, k_neighbors=5):
    if not isinstance(sampling_strategy, str):
        sampling_strategy = compute_sampling_strategy(sampling_strategy, Y_train, 'oversampling')
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
    print('Before SMOTE oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    print('After SMOTE oversampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def Borderline_SMOTE_os(X_train, Y_train, seed, sampling_strategy, k_neighbors=5):
    if not isinstance(sampling_strategy, str):
        sampling_strategy = compute_sampling_strategy(sampling_strategy, Y_train, 'oversampling')
    smote = BorderlineSMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
    print('Before Borderline SMOTE oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    print('After Borderline SMOTE oversampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def ADASYN_os(X_train, Y_train, seed, sampling_strategy, n_neighbors=5):
    if not isinstance(sampling_strategy, str):
        sampling_strategy = compute_sampling_strategy(sampling_strategy, Y_train, 'oversampling')
    sampling_strategy_copy = sampling_strategy
    adasyn = ADASYN(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, sampling_strategy=sampling_strategy)
    print('Before ADASYN oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = adasyn.fit_resample(X_train, Y_train)
    print('After ADASYN oversampling : ', sorted(Counter(Y_train_resampled).items()))
    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def RANDOM_os(X_train, Y_train, seed, sampling_strategy):
    if not isinstance(sampling_strategy, str):
        sampling_strategy = compute_sampling_strategy(sampling_strategy, Y_train, 'oversampling')
    ros = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)
    print('Before RANDOM oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)
    print('After RANDOM oversampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

################################################################################Undersampling Methods###############################################################################

def ENN_us(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, kind_sel='all'):
    enn = EditedNearestNeighbours(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    print('Before ENN undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = enn.fit_resample(X_train, Y_train)
    print('After ENN undersampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def Allknn_us(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, kind_sel='all'):
    knn = AllKNN(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    print('Before Allknn undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = knn.fit_resample(X_train, Y_train)
    print('After Allknn undersampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def Tomek_us(X_train, Y_train, seed, sampling_strategy):
    tl = TomekLinks(random_state=seed, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Tomek undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = tl.fit_resample(X_train, Y_train)
    print('After Tomek undersampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def RANDOM_us(X_train, Y_train, seed, sampling_strategy):
    if not isinstance(sampling_strategy, str):
        sampling_strategy = compute_sampling_strategy(sampling_strategy, Y_train,  'undersampling')
    print(sampling_strategy)
    ros = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)
    print('Before RANDOM undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)
    print('After RANDOM undersampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def CENTROID_us(X_train, Y_train, seed, sampling_strategy):
    if not isinstance(sampling_strategy, str):
        sampling_strategy = compute_sampling_strategy(sampling_strategy, Y_train,  'undersampling')
    cc = ClusterCentroids(random_state=seed, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Cluster Centroid undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = cc.fit_resample(X_train, Y_train)
    print('After Cluster Centroid undersampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def NearMiss_us(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, n_neighbors_ver3=3, version=1):
    if not isinstance(sampling_strategy, str):
        sampling_strategy = compute_sampling_strategy(sampling_strategy, Y_train,  'undersampling')
    nm = NearMiss(random_state=seed, version=version, n_neighbors=n_neighbors, n_neighbors_ver3=n_neighbors_ver3, n_jobs=-1, ratio=None, sampling_strategy=sampling_strategy)
    print('Before NearMiss version ' + str(version) + ' undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = nm.fit_resample(X_train, Y_train)
    print('After NearMiss version ' + str(version) + ' undersampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def IHT_us(X_train, Y_train, seed, sampling_strategy, estimator=None, cv=5):
    if not isinstance(sampling_strategy, str):
        sampling_strategy = compute_sampling_strategy(sampling_strategy, Y_train,  'undersampling')
    # Estimator can either be 'knn', 'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting' and 'linear-svm'
    iht = InstanceHardnessThreshold(estimator=estimator, random_state=seed, cv=cv, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Cluster Centroid undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = iht.fit_resample(X_train, Y_train)
    print('After Cluster Centroid undersampling : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

#############################################################Combination of Oversampling and Undersampling Methods#################################################################

def SMOTE_ENN(X_train, Y_train, seed, sampling_strategy, k_neighbors_smote=5, n_neighbors_enn=3, kind_sel='all'):
    enn = EditedNearestNeighbours(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors_enn, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors_smote, sampling_strategy=sampling_strategy)
    smote_enn = SMOTEENN(random_state=seed, smote=smote, enn=enn, sampling_strategy=sampling_strategy)
    print('Before SMOTE + ENN : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote_enn.fit_resample(X_train, Y_train)
    print('After SMOTE + ENN : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled

def SMOTE_Tomek(X_train, Y_train, seed, sampling_strategy, k_neighbors_smote=5):
    tl = TomekLinks(random_state=seed, n_jobs=-1)
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors_smote)
    smote_tomek = SMOTETomek(random_state=seed, smote=smote, tomek=tl)
    print('Before SMOTE + Tomek : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote_tomek.fit_resample(X_train, Y_train)
    print('After SMOTE + Tomek : ', sorted(Counter(Y_train_resampled).items()))

    X_train_resampled, Y_train_resampled = shuffle_dataset(X_train_resampled, Y_train_resampled, seed)

    return X_train_resampled, Y_train_resampled
