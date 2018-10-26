import os
import csv
import math
import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def report2csv(model_index, all_models_path, catalog_filename, constraints, ann_parameters, classification_problem, train_test_split, dataset_idx, cv_fold_nbr, others_flag, data_imputation, model_path, preprocessing, early_stopped_epoch, mean_auc_roc, mean_auc_pr, custom_metrics):

    csv_dict_inputs = {'model_index': [model_index],
                       'catalog': [catalog_filename],
                       'classification_problem': [classification_problem],
                       'constraints': [constraints],
                       'others_flag': [others_flag],
                       'data_imputation': [data_imputation],
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
                            'macro_precision': [custom_metrics.macro_precision],
                            'macro_recall': [custom_metrics.macro_recall],
                            'macro_f1score': [custom_metrics.macro_f1s],
                            'mean_auc_pr': [mean_auc_pr],
                            'mean_auc_roc': [mean_auc_roc]}

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

    return

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

def read_fits_as_dict(full_path):

    ####role####
    # Read a .fits file and returns the data and keys contained in the latter.

    ####inputs####
    # path : path of the .fits file to load
    # filename : filename of the .fits file to load

    ####outputs####
    # data_keys : The keys corresponding to the data inside the .fits file (eg DES_r, HSC_zphot, ...)
    # data : Object containing the data of the .fits file. The object has the same behavior than
    # python dictionnary (eg to access all the DES_r data just type data['DES_r'])

    hdu = fits.open(full_path)
    data = hdu[1].data
    data_keys = hdu[1].columns.names

    return data, data_keys

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

def compute_aucs(Y_pred, Y_val, X_val_id, classnames, savepath=None, plot=False):

    thresholds = list(np.linspace(0.00, 0.95, 20, endpoint=True))
    thresholds = [ round(elem, 2) for elem in thresholds ]
    fpr = {}
    tpr = {}
    ppv = {}
    auc_pr = []
    auc_roc = []
    nbr_classes = len(classnames)

    norm_fig = plt.figure(1, figsize=(19.2,12), dpi=100)
    norm_fig.subplots_adjust(left  = 0.1, bottom = 0.05, right = 0.9, top = 1.0, wspace = 0.9, hspace = 0.65)

    fig = plt.figure(2, figsize=(19.2,12), dpi=100)
    fig.subplots_adjust(left  = 0.1, bottom = 0.05, right = 0.9, top = 1.0, wspace = 0.9, hspace = 0.65)

    for idxplot, i in enumerate(thresholds.copy()):
        conf_matr, recall_correction = compute_conf_matr(Y_val, Y_pred, X_val_id, i, savepath)
        if conf_matr.shape[0] == nbr_classes:
            fpr, tpr, ppv = compute_roc_pr_curve(conf_matr, fpr, tpr, ppv, nbr_classes, recall_correction)
            if plot:
                plot_confusion_matrix(conf_matr, classnames, idxplot, i, tpr, ppv, savepath, True, 1)
                plot_confusion_matrix(conf_matr, classnames, idxplot, i, tpr, ppv, savepath, False, 2)
        else:
            thresholds.remove(i)

    fpr_roc = {}
    tpr_roc = {}
    tpr_pr = {}
    ppv_pr = {}

    for k in range(nbr_classes):
        idx_redordering = np.argsort(fpr[k])
        fpr_roc[k] = [fpr[k][h] for h in idx_redordering]
        tpr_roc[k] = [tpr[k][h] for h in idx_redordering]
        thresholds_roc = [thresholds[h] for h in idx_redordering]
        auc_roc.append(np.trapz(y=([0] + tpr_roc[k] + [1]), x=([0] + fpr_roc[k] + [1])))

    if plot:
        plt.close(1)
        plt.close(2)
        plot_roc_curve(fpr_roc, tpr_roc, ppv, thresholds_roc, nbr_classes, classnames, savepath, auc_roc)

    for k in range(nbr_classes):
        idx_redordering =  np.argsort(tpr[k])
        tpr_pr[k] = [tpr[k][h] for h in idx_redordering]
        ppv_pr[k] = [ppv[k][h] for h in idx_redordering]
        thresholds_pr = [thresholds[h] for h in idx_redordering]
        auc_pr.append(np.trapz(y=([1] + ppv_pr[k] + [0]), x=([0] + tpr_pr[k] + [1])))

    if plot:
        plt.close()
        plot_pr_curve(fpr, tpr_pr, ppv_pr, thresholds_pr, nbr_classes, classnames, savepath, auc_pr)

    mean_auc_pr = sum(auc_pr)/len(auc_pr)
    mean_auc_roc = sum(auc_roc)/len(auc_roc)

    return mean_auc_roc, mean_auc_pr

def compute_conf_matr(Y_val, Y_pred, X_val_id, threshold, savepath):

    unique, counts = np.unique(np.argmax(Y_val, axis=-1), return_counts=True)
    class_count_dict = dict(zip(unique, counts))
    nbr_classes = len(list(class_count_dict.keys()))
    Y_pred_non_category = []
    Y_val_non_category = []
    DES_id = []
    title = 'Threshold = ' + str(threshold)
    discard_count_dict = {}
    for i in list(class_count_dict.keys()):
        discard_count_dict[i] = 0
    recall_correction = {'macro': 0, 'micro': 0}
    for idxj, j in enumerate(Y_pred):
        if np.amax(j) >= threshold:
            Y_val_non_category.append(np.argmax(Y_val[idxj]))
            Y_pred_non_category.append(np.argmax(Y_pred[idxj]))
            DES_id.append(X_val_id[idxj])
        else:
            discard_count_dict[np.argmax(Y_val[idxj])] += 1

    for i in list(class_count_dict.keys()):
        recall_correction[i] = 1 - discard_count_dict[i]/class_count_dict[i]
        recall_correction['macro'] += recall_correction[i]

    recall_correction['macro'] = recall_correction['macro']/nbr_classes
    recall_correction['micro'] = 1 - sum(list(discard_count_dict.values()))/(sum(list(class_count_dict.values())))

    if savepath is not None:
        prediction_report = {'DES_id': DES_id, 'Y_true': Y_val_non_category, 'Y_pred': Y_pred_non_category}
        prediction_report_path, _ = os.path.split(savepath)
        pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(prediction_report_path, 'Predictions_' + str(threshold) + '.csv'), index=False)

    cm = confusion_matrix(Y_val_non_category, Y_pred_non_category)

    return cm, recall_correction

def compute_roc_pr_curve(confusion_matrix, fpr, tpr, ppv, n_classes, recall_correction):

    tpr['macro'] = 0.0
    ppv['macro'] = 0.0

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
        # print('class : ', i, ' FP : ', FP, ' FN : ', FN , ' TN : ', TN, ' TP : ', TP)
        if (i in list(tpr.keys())) and (i in list(fpr.keys())) and (i in list(ppv.keys())):
            tpr[i].append(TP/(TP+FN)*recall_correction[i])
            fpr[i].append(FP/(FP+TN))
            ppv[i].append(TP/(FP+TP))
        else:
            tpr[i] = [TP/(TP+FN)*recall_correction[i]]
            fpr[i] = [FP/(FP+TN)]
            ppv[i] = [TP/(FP+TP)]
        tpr['macro'] += tpr[i][-1]
        ppv['macro'] += ppv[i][-1]

    tpr['macro'] = tpr['macro']/n_classes
    ppv['macro'] = ppv['macro']/n_classes

    return fpr, tpr, ppv

def plot_confusion_matrix(cm, target_names, idx_plot, threshold, tpr, ppv, savepath, normalize, figure_index):

    plt.figure(figure_index)
    title='Threshold = {:0.2f}'.format(threshold)

    macro_recall = tpr['macro']
    macro_precision = ppv['macro']

    cmap = plt.get_cmap('Blues')

    plt.subplot(4, 5, idx_plot + 1)
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="yellow" if cm[i, j] > thresh else "black", fontsize=6)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="yellow" if cm[i, j] > thresh else "black", fontsize=6)


    # plt.tight_layout()
    plt.ylabel(title + '\nTrue label')
    plt.xlabel('[macro] recall={:0.2f}; precision={:0.2f}'.format(macro_recall, macro_precision))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if normalize:
        plt.savefig(os.path.join(savepath, 'cm_norm.png'))
    else:
        plt.savefig(os.path.join(savepath, 'cm.png'))

    return

def plot_roc_curve(fpr, tpr, ppv, thresholds, n_classes, classnames, savepath, auc_roc):

    colors = ['blue', 'brown', 'cyan', 'red', 'yellow', 'magenta', 'lime', 'orange', 'purple', 'lightgray', 'gold', 'darkblue', 'tan', 'olive', 'turquoise', 'pink', 'darkgreen', 'gray', 'lightsalmon', 'black']

    thresholds_copy = thresholds.copy()
    colors_copy = colors.copy()
    for i in range(n_classes):

        plt.figure(figsize=(19.2,10.8), dpi=100)
        for idxj, j in enumerate(thresholds_copy):
            plt.scatter(fpr[i][idxj], tpr[i][idxj], color=colors_copy[idxj],
                     label='Threshold = {0})'
                     ''.format(thresholds_copy[idxj]))
        plt.step([0] + fpr[i] + [1],[0] + tpr[i] + [1], linestyle=':', lw=1, color='black')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Area Under Curve of class {} : {:0.2f}'.format(classnames[i], auc_roc[i]))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(savepath, 'roc_' + classnames[i] + '.png'))
        plt.close()

    return

def plot_pr_curve(fpr, tpr, ppv, thresholds, n_classes, classnames, savepath, auc_pr):

    colors = ['blue', 'brown', 'cyan', 'red', 'yellow', 'magenta', 'lime', 'orange', 'purple', 'lightgray', 'gold', 'darkblue', 'tan', 'olive', 'turquoise', 'pink', 'darkgreen', 'gray', 'lightsalmon', 'black']

    thresholds_copy = thresholds.copy()
    colors_copy = colors.copy()
    for i in range(n_classes):

        plt.figure(figsize=(19.2,10.8), dpi=100)
        for idx in range(len(thresholds_copy)):
            plt.scatter(tpr[i][idx], ppv[i][idx], color=colors_copy[idx],
                     label='Threshold = {0})'
                     ''.format(thresholds_copy[idx]))
        plt.step([0] + tpr[i] + [1],[0] + ppv[i] + [0], linestyle=':' ,lw=1, color='black')
        plt.plot([0, 1], [0, 0], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Area Under Curve of class {} : {:0.2f}'.format(classnames[i], auc_pr[i]))
        plt.legend(loc="best")
        plt.savefig(os.path.join(savepath, 'pr_curve_' + classnames[i] + '.png'))
        plt.close()

    return

def csv2dict_list(csv_path):

    # Open variable-based csv, iterate over the rows and map values to a list of dictionaries containing key/value pairs

    with open(csv_path) as f:
        dict_list = [ {k:v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    print(type(dict_list[0]['train_test_split / train_val_split']))
    return dict_list

def format_csv_dict(model_input):

    if model_input['preprocessing']:
        model_input_preprocessing_method = []
        model_input['preprocessing'] = (model_input['preprocessing'][1:-1]).split(',')
        model_input['preprocessing_params'] = (model_input['preprocessing'][1:-1]).split(',')
        for i in range(len(model_input['preprocessing'])):
            model_input_preprocessing.append({'method': model_input['preprocessing'][i], 'arguments': model_input['preprocessing_params'][i]})
        model_input['preprocessing_method'] = model_input_preprocessing_method
    else:
        model_input['preprocessing_method'] = None

    model_input_training_testing_split = []
    model_input['train_test_split / train_val_split'] = (model_input['train_test_split / train_val_split'][1:-1]).split(',')
    for i in range(len(model_input['train_test_split / train_val_split'])):
        model_input_training_testing_split.append(int(model_input['train_test_split / train_val_split'][i]))
    model_input['training_testing_split'] = model_input_training_testing_split

    model_input['dataset_idx'] = int(model_input['dataset_idx'])
    model_input['cv_fold_nbr'] = int(model_input['cv_fold_nbr'])
    model_input['data_imputation'] = float(model_input['data_imputation'])

    return model_input

def get_model_weights_path(model_path, weights_flag):

    for file in os.listdir(model_path):
        if file.endswith(".hdf5"):
            final_model_weights = os.path.join(model_path, file)
            final_model_score = float((file.split('-')[1]).split('_')[-1])
            if weights_flag == 'final':
                return final_model_weights

    for file in os.listdir(os.path.join(model_path, 'checkpoints')):
        if file.endswith(".hdf5"):
            checkpoint_model_weights = os.path.join(model_path, 'checkpoints', file)
            checkpoint_score = float((file.split('-')[1]).split('_')[-1])
            if weights_flag == 'checkpoint':
                return checkpoint_model_weights

    if weights_flag == 'best':
        if checkpoint_score > final_model_score:
            return checkpoint_model_weights
        else:
            return final_model_weights

    return

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
