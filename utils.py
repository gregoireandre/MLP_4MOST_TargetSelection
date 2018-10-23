import os
import math
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.metrics import auc, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

#########General utilities functions################

def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = {}
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            if class_label not in list(D_class_data.keys()):
                D_class_data[class_label] = {}
                D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
            else:
                D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data

def report2csv(reportdict, catalog_filename, constraints, ann_parameters, classification_problem, train_test_split, dataset_idx, cv_fold_nbr, others_flag, model_path, preprocessing, early_stopped_epoch, mean_auc_roc, mean_auc_pr):

    noothers_precision = []
    noothers_recall = []
    noothers_f1_score = []
    noothers_precision_density = []

    for i in list(reportdict.keys()):
        if ('Others' not in i) and ('avg' not in i):
            noothers_precision.append(reportdict[i]['precision'])
            noothers_recall.append(reportdict[i]['recall'])
            noothers_f1_score.append(reportdict[i]['f1-score'])
        elif ('Others' in i):
            others_precision = reportdict[i]['precision']
            others_recall = reportdict[i]['recall']
            others_f1_score = reportdict[i]['f1-score']

    noothers_mean_precision = sum(noothers_precision)/len(noothers_precision)
    noothers_mean_recall = sum(noothers_recall)/len(noothers_recall)
    noothers_mean_f1_score = sum(noothers_f1_score)/len(noothers_f1_score)
    all_models_path, model_index = os.path.split(model_path)

    csv_dict_inputs = {'model_index': [model_index],
                       'catalog': [catalog_filename],
                       'classification_problem': [classification_problem],
                       'constraints': [constraints],
                       'others_flag': [others_flag],
                       'train_test_split / train_val_split': [train_test_split],
                       'dataset_idx': [dataset_idx],
                       'cv_fold_nbr': [cv_fold_nbr],
                       'preprocessing': [],
                       'preprocessing_params': []}
    if preprocessing is not None:
        for i in preprocessing:
            csv_dict_inputs['preprocessing'].append(i['method'])
            csv_dict_inputs['preprocessing_params'].append(i['arguments'])
    else:
        csv_dict_inputs['preprocessing'].append(None)
        csv_dict_inputs['preprocessing_params'].append(None)

    csv_dict_performance = {'model_index': [model_index],
                            'early_stop': [early_stopped_epoch],
                            'auc_pr': [mean_auc_pr],
                            'auc_roc': [mean_auc_roc],
                            'precision': [noothers_mean_precision],
                            'recall': [noothers_mean_recall],
                            'f1_score': [noothers_mean_f1_score],
                            'others_precision': [others_precision],
                            'others_recall': [others_recall],
                            'others_f1_score': [others_f1_score]}

    csv_dict_parameters = {'model_index': [model_index]}

    for i in list(ann_parameters.keys()):
        if (i != 'optimizer') and (i != 'kernel_regularizer'):
            csv_dict_parameters[i] = [ann_parameters[i]]

    benchmark_filename = os.path.join(all_models_path, 'Benchmark_ANN_inputs.csv')

    if os.path.isfile(benchmark_filename):
        csv_input = pd.read_csv(benchmark_filename)
        new_csv = pd.concat([csv_input, pd.DataFrame.from_dict(csv_dict_inputs)], sort=False)
        new_csv.to_csv(benchmark_filename, index=False)
    else:
        pd.DataFrame.from_dict(csv_dict_inputs).to_csv(benchmark_filename, index=False)

    benchmark_filename = os.path.join(all_models_path, 'Benchmark_ANN_parameters.csv')

    if os.path.isfile(benchmark_filename):
        csv_input = pd.read_csv(benchmark_filename)
        new_csv = pd.concat([csv_input, pd.DataFrame.from_dict(csv_dict_parameters)])
        new_csv.to_csv(benchmark_filename, index=False)
    else:
        pd.DataFrame.from_dict(csv_dict_parameters).to_csv(benchmark_filename, index=False)

    benchmark_filename = os.path.join(all_models_path, 'Benchmark_ANN_performance.csv')

    if os.path.isfile(benchmark_filename):
        csv_input = pd.read_csv(benchmark_filename)
        new_csv = pd.concat([csv_input, pd.DataFrame.from_dict(csv_dict_performance)])
        new_csv.to_csv(benchmark_filename, index=False)
    else:
        pd.DataFrame.from_dict(csv_dict_performance).to_csv(benchmark_filename, index=False)

def read_fits(full_path):

    hdu = fits.open(full_path)
    data = hdu[1].data
    data_keys = hdu[1].columns.names

    nbr_features = len(data_keys)
    nbr_objects = len(data)
    data_array = np.zeros((nbr_objects, nbr_features), dtype=np.float64)

    for i in range(nbr_features):
        data_array[:,i] = data.field(i)

    return data_array

def constraints_to_str(constraints):

    if constraints == 'no':
        all_constraints_str = constraints
    else:
        all_constraints_str = ''
        for s in list(constraints.keys()):
            if all_constraints_str:
                all_constraints_str += '+' + s
            else:
                all_constraints_str = s

    return all_constraints_str

def compute_classnames(classification_problem, others_flag):

    if others_flag == 'no':
        if 'binary' in classification_problem:
            classnames = ['Others', classification_problem.split('_')[0]]
        elif classification_problem == 'BG_ELG_LRG_classification':
            classnames = ['ELG', 'LRG', 'BG']
        elif classification_problem == 'BG_ELG_LRG_QSO_classification':
            classnames = ['ELG', 'LRG', 'BG', 'QSO']
        elif classification_problem == 'BG_LRG_QSO_classification':
            classnames = ['LRG', 'BG', 'QSO']
    elif others_flag == 'all':
        if 'binary' in classification_problem:
            classnames = ['Others', classification_problem.split('_')[0]]
        elif classification_problem == 'BG_ELG_LRG_classification':
            classnames = ['Others', 'ELG', 'LRG', 'BG']
        elif classification_problem == 'BG_ELG_LRG_QSO_classification':
            classnames = ['Others', 'ELG', 'LRG', 'BG', 'QSO']
        elif classification_problem == 'BG_LRG_QSO_classification':
            classnames = ['Others', 'LRG', 'BG', 'QSO']

    return classnames

def compute_aucs(Y_pred, Y_val, classnames):

    thresholds = list(np.linspace(0.05, 0.95, 19, endpoint=True))
    thresholds = [ round(elem, 2) for elem in thresholds ]
    fpr = {}
    tpr = {}
    ppv = {}
    auc_pr = []
    auc_roc = []

    for  i in thresholds.copy():
        conf_matr, nbr_classes = compute_conf_matr(Y_val, Y_pred, i, classnames)
        if conf_matr.shape[0] == nbr_classes:
            fpr, tpr, ppv = compute_roc_curve(conf_matr, fpr, tpr, ppv, nbr_classes)
        else:
            thresholds.remove(i)

    for i in list(fpr.keys()):
        auc_pr.append(np.trapz(tpr[i], ppv[i]))
        auc_roc.append(np.trapz(fpr[i], tpr[i]))

    mean_auc_pr = sum(auc_pr)/len(auc_pr)
    mean_auc_roc = sum(auc_roc)/len(auc_roc)

    return mean_auc_roc, mean_auc_pr

def compute_conf_matr(Y_val, Y_pred, threshold, classnames):

    n_classes = len(classnames)
    y_pred_non_category = []
    Y_val_non_category = []
    title = 'Threshold = ' + str(threshold)
    for idxj, j in enumerate(Y_pred):
        if np.amax(j) >= threshold:
            Y_val_non_category.append(np.argmax(Y_val[idxj]))
            y_pred_non_category.append(np.argmax(Y_pred[idxj]))
    cm = confusion_matrix(Y_val_non_category, y_pred_non_category)

    return cm, n_classes

    return

def compute_roc_curve(confusion_matrix, fpr, tpr, ppv, n_classes):

    # Plot linewidth.
    lw = 2

    for i in range(n_classes):
        TP = confusion_matrix[i][i]
        FP = 0
        TN = 0
        FN = 0
        for j in range(n_classes):
            if j != i:
                FP += confusion_matrix[j][i]
                FN += confusion_matrix[i][j]
                TN += confusion_matrix[j][j]
        print('class : ', i, ' FP : ', FP, ' FN : ', FN , ' TN : ', TN, ' TP : ', TP)
        if (i in list(tpr.keys())) and (i in list(fpr.keys())):
            tpr[i].append(TP/(TP+FN))
            fpr[i].append(FP/(FP+TN))
            ppv[i].append(TP/(FP+TP))
        else:
            tpr[i] = [TP/(TP+FN)]
            fpr[i] = [FP/(FP+TN)]
            ppv[i] = [TP/(FP+TP)]
    return fpr, tpr, ppv

#######Dataset processing methods#########

def load_dataset(dataset_path, classification_problem, train_test_split, dataset_idx, cv_fold_nbr):

    training_dataset_filename = classification_problem + '_train_' + str(train_test_split[0]) + '_' + str(train_test_split[1]) + '_' + str(dataset_idx) + '_' + str(cv_fold_nbr) + '.fits'
    validation_dataset_filename = classification_problem + '_val_' + str(train_test_split[0]) + '_' + str(train_test_split[1]) + '_' + str(dataset_idx) + '_' + str(cv_fold_nbr) + '.fits'
    testing_dataset_filename = classification_problem + '_test_' + str(train_test_split[0]) + '_' + str(train_test_split[1]) + '_' + str(dataset_idx) + '.fits'

    # load  dataset
    training_dataset = read_fits(os.path.join(dataset_path, training_dataset_filename))
    np.random.shuffle(training_dataset)
    validation_dataset = read_fits(os.path.join(dataset_path, validation_dataset_filename))
    np.random.shuffle(validation_dataset)
    testing_dataset = read_fits(os.path.join(dataset_path, testing_dataset_filename))
    np.random.shuffle(testing_dataset)

    # split into input (X) and output (Y) variables
    X_train = training_dataset[:,1:-1]
    Y_train = training_dataset[:,-1]

    # split into input (X) and output (Y) variables
    DES_id_val = validation_dataset[:,0]
    X_val = validation_dataset[:,1:-1]
    Y_val = validation_dataset[:,-1]

    # split into input (X) and output (Y) variables
    DES_id_test = validation_dataset[:,0]
    X_test = testing_dataset[:,:-1]
    Y_test = testing_dataset[:,-1]

    return X_train, Y_train, X_val, Y_val, DES_id_val, X_test, Y_test, DES_id_test

def compute_weights(Y):

    class_weights = compute_class_weights(Y)
    sample_weights = compute_sample_weights(Y, class_weights)

    return sample_weights

def compute_sample_weights(Y, class_weights):

    unique, counts = np.unique(Y, return_counts=True)
    nbr_objects = Y.shape[0]
    class_count_dict = dict(zip(unique, counts))
    print(class_count_dict)
    all_labels = list(class_count_dict.keys())
    sample_weights = []

    for i in Y:
        for j in all_labels:
            if i == j:
                sample_weights.append(class_weights[j])

    return np.array(sample_weights)

def compute_class_weights(Y):
    unique, counts = np.unique(Y, return_counts=True)
    nbr_objects = Y.shape[0]
    class_count_dict = dict(zip(unique, counts))
    labels = list(class_count_dict.keys())
    nbr_labels = len(labels)
    class_weights = {}
    for i in labels:
        class_weights[i] = nbr_objects/(len(labels)*class_count_dict[i])

    return class_weights

def one_hot_encode(Y_train, Y_val, Y_test):

    Y_train = np.array(Y_train, dtype='int')
    Y_train = Y_train.reshape(len(Y_train), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_train = onehot_encoder.fit_transform(Y_train)

    Y_val = np.array(Y_val, dtype='int')
    Y_val = Y_val.reshape(len(Y_val), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_val = onehot_encoder.fit_transform(Y_val)

    Y_test = np.array(Y_test, dtype='int')
    Y_test = Y_test.reshape(len(Y_test), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_test = onehot_encoder.fit_transform(Y_test)

    return Y_train, Y_val, Y_test

def reduce_class_ratio(dataset, label, class_fraction):


    class_idx = []
    non_class_count = 0
    non_class_idx = []
    for idx, i in enumerate(dataset[:,-1]) :
        if i == label:
            class_idx.append(idx)
        else:
            non_class_count += 1
            non_class_idx.append(idx)
    nbr_class_to_select = int(class_fraction*len(class_idx))
    np.random.shuffle(class_idx)
    class_idx_to_select = class_idx[:nbr_class_to_select]
    all_objects_to_keep = sorted(non_class_idx + class_idx_to_select)
    reduced_dataset = dataset[all_objects_to_keep, :]

    return reduced_dataset

def compute_others_at_border(training_dataset):
    unique, counts = np.unique(training_dataset[:, -1], return_counts=True)
    nbr_objects = training_dataset.shape[0]
    class_dict = dict(zip(unique, counts))
    all_class_objects = {}
    all_class_idx = {}
    all_centroids = {}
    isclass = {}
    others_neighbors_idx = {}
    all_indexes_to_keep = []
    for label in list(class_dict.keys()):
            isclass[label] = (training_dataset[:,-1]==label)
            all_class_idx[label] = np.where(isclass[label])[0]
            all_class_objects[label] = training_dataset[all_class_idx[label],:]
            all_centroids[label] = np.average(all_class_objects[label][1:-1], axis=0)
    tree = cKDTree(all_class_objects[0.0])
    for label in list(class_dict.keys()):
        if label != 0.0:
            _, others_neighbors_idx[label] = tree.query(all_centroids[label], k=(class_dict[label]), n_jobs=-1)
    for label in list(class_dict.keys()):
        if label == 0.0:
            for labelbis in list(class_dict.keys()):
                if labelbis != 0.0:
                    all_indexes_to_keep = list(all_indexes_to_keep) + list(all_class_idx[label][others_neighbors_idx[labelbis]])
        else:
            all_indexes_to_keep = list(all_indexes_to_keep) + list(all_class_idx[label])

    reduced_dataset = training_dataset[all_indexes_to_keep,:]

    return reduced_dataset
