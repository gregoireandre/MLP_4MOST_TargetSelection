# Module that contains various preprocessing techniques for imbalanced datasets.
# The documentation for those functions can be found at : https://imbalanced-learn.readthedocs.io/en/stable/index.html

import os
import numpy as np
import pandas as pd
from collections import Counter
from astropy.io import fits
from imblearn.ensemble import BalanceCascade
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler, EditedNearestNeighbours, TomekLinks, AllKNN, InstanceHardnessThreshold

def save_preprocessed_dataset(script_dir, catalog_filename, constraints, data_imputation, normalization, classification_problem, train_test_val_split, cv_fold_nbr, preprocessing_methods, X_train, Y_train):

    dataset_path = os.path.join(script_dir, 'datasets', catalog_filename, constraints + '-constraints', classification_problem + '_train_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '_' + str(cv_fold_nbr) + '.fits')
    savepath = os.path.join(script_dir, 'datasets', catalog_filename, constraints + '-constraints', 'preprocessed_' + str(data_imputation) + '-imputation', classification_problem + '_train_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '_' + str(cv_fold_nbr) + '_norm-' + str(normalization))
    for idx, i in enumerate(preprocessing_methods):
        if idx == 0:
            filename = i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])
        else:
            filename += '_' + i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    table_dataset_train = np.column_stack((X_train, Y_train))

    hdu = fits.open(dataset_path)
    reduced_keys = hdu[1].columns.names
    reduced_keys.pop(0)

    all_fits_column = []
    for g in range(table_dataset_train.shape[1]):
        all_fits_column.append(fits.Column(name=reduced_keys[g], array=table_dataset_train[:,g], format='D'))
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

# def remove_imputed_datapoints(X, ):
#
#     imputation_mask = (X_train == imputation_value)
#     imputation_idx = np.where(imputation_mask)[0]
#     imputation_keys = np.where(imputation_mask)[1]
#     no_imputation_idx = np.where(not imputation_mask)[0]
#     no_imputation_keys = np.where(not imputation_mask)[1]
#     X_train_no_imputation = X_train[no_imputation_idx,:]
#     X_train_imputation = X_train[imputation_idx,no_imputation_keys]
#
#
#     return


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
