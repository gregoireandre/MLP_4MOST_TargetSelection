# Module that contains various preprocessing techniques for imbalanced datasets.
# The documentation for those functions can be found at : https://imbalanced-learn.readthedocs.io/en/stable/index.html

import os
import numpy as np
import pandas as pd
from collections import Counter
from astropy.io import fits
from imblearn.ensemble import BalanceCascade
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler, EditedNearestNeighbours, TomekLinks, AllKNN, InstanceHardnessThreshold

def save_preprocessed_dataset(script_dir, catalog_filename, others_flag, constraints, data_imputation, classification_problem, train_test_split, dataset_idx, cv_fold_nbr, preprocessing_methods, X_train, Y_train):

    dataset_path = os.path.join(script_dir, 'datasets', catalog_filename,  others_flag + '-others_' + constraints + '-constraints_' + str(data_imputation) + '-imputation', classification_problem + '_train_' + str(train_test_split[0]) + '_' + str(train_test_split[1]) + '_' + str(dataset_idx) + '_' + str(cv_fold_nbr) + '.fits')
    savepath = os.path.join(script_dir, 'datasets', catalog_filename,  others_flag + '-others_' + constraints + '-constraints_' + str(data_imputation) + '-imputation', 'preprocessed', classification_problem + '_train_' + str(train_test_split[0]) + '_' + str(train_test_split[1]) + '_' + str(dataset_idx) + '_' + str(cv_fold_nbr))
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

################################################################################Oversampling Methods###############################################################################

def SMOTE_oversampling(X_train, Y_train, seed, sampling_strategy, k_neighbors=5):
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
    print('Before SMOTE oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    print('After SMOTE oversampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def ADASYN_oversampling(X_train, Y_train, seed, sampling_strategy, n_neighbors=5):
    adasyn = ADASYN(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, sampling_strategy=sampling_strategy)
    print('Before ADASYN oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = adasyn.fit_resample(X_train, Y_train)
    print('After ADASYN oversampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def RANDOM_oversampling(X_train, Y_train, seed, sampling_strategy):
    ros = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)
    print('Before RANDOM oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)
    print('After RANDOM oversampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

################################################################################Undersampling Methods###############################################################################

def ENN_undersampling(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, kind_sel='all'):
    enn = EditedNearestNeighbours(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    print('Before ENN undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = enn.fit_resample(X_train, Y_train)
    print('After ENN undersampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def Allknn_undersampling(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, kind_sel='all'):
    knn = AllKNN(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    print('Before Allknn undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = knn.fit_resample(X_train, Y_train)
    print('After Allknn undersampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def Tomek_undersampling(X_train, Y_train, seed, sampling_strategy):
    tl = TomekLinks(random_state=seed, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Tomek undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = tl.fit_resample(X_train, Y_train)
    print('After Tomek undersampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def RANDOM_undersampling(X_train, Y_train, seed, sampling_strategy):
    ros = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)
    print('Before RANDOM undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)
    print('After RANDOM undersampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def CENTROID_undersampling(X_train, Y_train, seed, sampling_strategy):

    cc = ClusterCentroids(random_state=seed, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Cluster Centroid undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = cc.fit_resample(X_train, Y_train)
    print('After Cluster Centroid undersampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def NearMiss_undersampling(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, n_neighbors_ver3=3, version=1):

    nm = NearMiss(random_state=seed, version=version, n_neighbors=n_neighbors, n_neighbors_ver3=n_neighbors_ver3, n_jobs=-1, ratio=None, sampling_strategy=sampling_strategy)
    print('Before NearMiss version ' + str(version) + ' undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = nm.fit_resample(X_train, Y_train)
    print('After NearMiss version ' + str(version) + ' undersampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def IHT_undersampling(X_train, Y_train, seed, sampling_strategy, estimator='adaboost', cv=5):

    # Estmator can either be 'knn', 'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting' and 'linear-svm'

    iht = InstanceHardnessThreshold(estimator=estimator, random_state=seed, cv=cv, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Cluster Centroid undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = cc.fit_resample(X_train, Y_train)
    print('After Cluster Centroid undersampling : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

#############################################################Combination of Oversampling and Undersampling Methods#################################################################

def SMOTE_ENN(X_train, Y_train, seed, sampling_strategy, k_neighbors_smote=5, n_neighbors_enn=3, kind_sel='all'):
    enn = EditedNearestNeighbours(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors_enn, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors_smote, sampling_strategy=sampling_strategy)
    smote_enn = SMOTEENN(random_state=seed, smote=smote, enn=enn, sampling_strategy=sampling_strategy)
    print('Before SMOTE + ENN : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote_enn.fit_resample(X_train, Y_train)
    print('After SMOTE + ENN : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled

def SMOTE_Tomek(X_train, Y_train, seed, sampling_strategy, k_neighbors_smote=5):
    tl = TomekLinks(random_state=seed, n_jobs=-1)
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors_smote)
    smote_tomek = SMOTETomek(random_state=seed, smote=smote, tomek=tl)
    print('Before SMOTE + Tomek : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote_tomek.fit_resample(X_train, Y_train)
    print('After SMOTE + Tomek : ', sorted(Counter(Y_train_resampled).items()))

    return X_train_resampled, Y_train_resampled
