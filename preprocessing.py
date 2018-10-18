import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler, EditedNearestNeighbours, TomekLinks, AllKNN, InstanceHardnessThreshold
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.ensemble import BalanceCascade

def reconstruct_dataset(X_train_resampled, Y_train_resampled):
    dataset = np.zeros((X_train_resampled.shape[0], X_train_resampled.shape[1] + 1))
    dataset[:, :-1] = X_train_resampled
    dataset[:, -1] = Y_train_resampled
    return dataset

#####Oversampling methods#####

def SMOTE_oversampling(X_train, Y_train, seed, sampling_strategy, k_neighbors=5):
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
    print('Before SMOTE oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    print('After SMOTE oversampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def ADASYN_oversampling(X_train, Y_train, seed, sampling_strategy, n_neighbors=5):
    adasyn = ADASYN(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, sampling_strategy=sampling_strategy)
    print('Before ADASYN oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = adasyn.fit_resample(X_train, Y_train)
    print('After ADASYN oversampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def RANDOM_oversampling(X_train, Y_train, seed, sampling_strategy):
    ros = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)
    print('Before RANDOM oversampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)
    print('After RANDOM oversampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

#####Undersampling methods#####

def ENN_undersampling(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, kind_sel='all'):
    enn = EditedNearestNeighbours(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    print('Before ENN undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = enn.fit_resample(X_train, Y_train)
    print('After ENN undersampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def Allknn_undersampling(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, kind_sel='all'):
    knn = AllKNN(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    print('Before Allknn undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = knn.fit_resample(X_train, Y_train)
    print('After Allknn undersampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def Tomek_undersampling(X_train, Y_train, seed, sampling_strategy):
    tl = TomekLinks(random_state=seed, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Tomek undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = tl.fit_resample(X_train, Y_train)
    print('After Tomek undersampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def RANDOM_undersampling(X_train, Y_train, seed, sampling_strategy):
    ros = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)
    print('Before RANDOM undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)
    print('After RANDOM undersampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def CENTROID_undersampling(X_train, Y_train, seed, sampling_strategy):

    cc = ClusterCentroids(random_state=seed, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Cluster Centroid undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = cc.fit_resample(X, y)
    print('After Cluster Centroid undersampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def NearMiss_undersampling(X_train, Y_train, seed, sampling_strategy, n_neighbors=3, n_neighbors_ver3=3, version=1):

    nm = NearMiss(random_state=seed, version=version, n_neighbors=n_neighbors, n_neighbors_ver3=n_neighbors_ver3, n_jobs=1, ratio=None, sampling_strategy=sampling_strategy)
    print('Before NearMiss version ' + str(version) + ' undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = nm.fit_resample(X, y)
    print('After NearMiss version ' + str(version) + ' undersampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def IHT_undersampling(X_train, Y_train, seed, sampling_strategy, estimator='adaboost', cv=5):

    # Estmator can either be 'knn', 'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting' and 'linear-svm'

    iht = InstanceHardnessThreshold(estimator=estimator, random_state=seed, cv=cv, n_jobs=-1, sampling_strategy=sampling_strategy)
    print('Before Cluster Centroid undersampling : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = cc.fit_resample(X, y)
    print('After Cluster Centroid undersampling : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

#####Combination of Undersampling and Oversampling methods#####

def SMOTE_ENN(X_train, Y_train, seed, sampling_strategy, k_neighbors_smote=5, n_neighbors_enn=3, kind_sel='all'):
    enn = EditedNearestNeighbours(random_state=seed, n_jobs=-1, n_neighbors=n_neighbors_enn, kind_sel=kind_sel, sampling_strategy=sampling_strategy)
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors_smote, sampling_strategy=sampling_strategy)
    smote_enn = SMOTEENN(random_state=seed, smote=smote, enn=enn, sampling_strategy=sampling_strategy)
    print('Before SMOTE + ENN : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote_enn.fit_resample(X_train, Y_train)
    print('After SMOTE + ENN : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled

def SMOTE_Tomek(X_train, Y_train, seed, sampling_strategy, k_neighbors_smote=5):
    tl = TomekLinks(random_state=seed, n_jobs=-1)
    smote = SMOTE(random_state=seed, n_jobs=-1, k_neighbors=k_neighbors_smote)
    smote_tomek = SMOTETomek(random_state=seed, smote=smote, tomek=tl)
    print('Before SMOTE + Tomek : ', sorted(Counter(Y_train).items()))
    X_train_resampled, Y_train_resampled = smote_tomek.fit_resample(X_train, Y_train)
    print('After SMOTE + Tomek : ', sorted(Counter(Y_train_resampled).items()))

    dataset_resampled = reconstruct_dataset(X_train_resampled, Y_train_resampled)
    return dataset_resampled
