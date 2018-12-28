import os
import csv
import time
import math
import copy
import pprint
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import Color
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from sklearn.metrics import auc, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, Normalizer

#########General utilities functions################

def report2csv(model_index, all_models_path, catalog_filename, constraints, ann_parameters, classification_problem, train_test_val_split, cv_fold_nbr, data_imputation, normalization, model_path, preprocessing, early_stopped_epoch, mean_auc_roc, mean_auc_pr, custom_metrics):

    ####role####
    # Store all informations regarding :   - The inputs given to the CLassification Object of classification.py
    #                                      - The parameters of the neural network for this Classification object
    #                                      - The performances of the neural network
    # All those informations are appended to a .csv file that is common for all runs
    # This allows effective comparison of performances between different models and makes it easier to understand which parameters gives best performances

    ####inputs####
    # model_index (int) : The index of the model defined in classification.py
    # all_models_path (string) : The path under which all the models performances and weights are stored
    # catalog_filename (string) : The filename of the catalog used.
    # constraints (string) : The constraints flag corresponding to the dataset used
    # ann_parameters (dictionnary) : A dictionnary containing all the ANN parameters (see classification.py)
    # classification_problem (string) : A string for the classification problem considered (i.e 'BG_LRG_ELG_QSO_classification')
    # train_test_val_split (list) : A list containing the percentage of the dataset used for training, validation and testing (i.e [perc_train, perc_val, perc_test])
    #                               Note that the percentage for validation is a percentage relative to the training set and not the whole dataset.
    #                               For example : [80, 20, 20] means that 80% of the dataset is used as training
    # cv_fold_nbr (int) : the fold number of the dataset
    # data_imputation (string or float) : If float, the value used for data imputation.
    #                                     If string, the method used for data imoutation (i.e 'mean, 'max')
    # normalization (string) : The normalization method used as preprocessing (i.e 'cr', 'quant')
    # model_path (string) : The path to the model folder
    # preprocessing (list of dictionnaries) : As described in classification.py a preprocessing method is stored as a dictionnary following : {'method': method_string, 'arguments': [aruments_for_method_1, ..., aruments_for_method_n]}.
    #                                         The method and arguments refers to the preprocessing_classification.py file.
    #                                         To handle the case where several preprocessing method are used, the general form of preprocessing_methods is a list of dict.
    #
    # early_stopped_epoch (int) : The epochs at which the early stop occured if the option was enabled
    # mean_auc_roc (float) : The mean of the Area Under Curve for ROC curves
    # mean_auc_pr (float) : The mean of the Area Under Curve for PR curves
    # custom_metrics (custom metrics class object) : Custom metrics have attributes that gives more in-deptth treview of performances for each classes (see custom_metrics.py for more info)

    # The parameters of the model are appended to the .csv using pandas dataframe.from_dict function
    # From the nature of appending data in python we need to use list in each dictionnary entry even if the list only contains a single element

    csv_dict_inputs = {'model_index': [model_index],
                       'catalog': [catalog_filename],
                       'classification_problem': [classification_problem],
                       'constraints': [constraints],
                       'data_imputation': [data_imputation],
                       'normalization': [normalization],
                       'train_test_val_split': [train_test_val_split],
                       'cv_fold_nbr': [cv_fold_nbr],
                       'preprocessing': [],
                       'preprocessing_params': []
                      }

    if preprocessing is not None:
        for idx, i in enumerate(preprocessing):
            if idx == 0:
                csv_dict_inputs['preprocessing'].append(i['method'])
                csv_dict_inputs['preprocessing_params'].append(str(i['arguments']))
            else:
                csv_dict_inputs['preprocessing'][0] += '___' + i['method']
                csv_dict_inputs['preprocessing_params'][0] += '___' + str(i['arguments'])
    else:
        csv_dict_inputs['preprocessing'].append(None)
        csv_dict_inputs['preprocessing_params'].append(None)

    csv_dict_performance = {'model_index': [model_index],
                            'early_stop': [early_stopped_epoch],
                            'macro_precision': [custom_metrics.macro_precision[-1]],
                            'macro_recall': [custom_metrics.macro_recall[-1]],
                            'macro_f1score': [custom_metrics.macro_f1s[-1]],
                            'mean_auc_pr': [mean_auc_pr],
                            'mean_auc_roc': [mean_auc_roc]
                           }
    for i in list(custom_metrics.others.keys()):
        csv_dict_performance['Others_' + i] = [custom_metrics.others[i][-1]]
    for i in list(custom_metrics.bgs.keys()):
        csv_dict_performance['BG_' + i] = [custom_metrics.bgs[i][-1]]
    for i in list(custom_metrics.lrgs.keys()):
        csv_dict_performance['LRG_' + i] = [custom_metrics.lrgs[i][-1]]
    for i in list(custom_metrics.elgs.keys()):
        csv_dict_performance['ELG_' + i] = [custom_metrics.elgs[i][-1]]
    for i in list(custom_metrics.qsos.keys()):
        csv_dict_performance['QSO_' + i] = [custom_metrics.qsos[i][-1]]

    csv_dict_parameters = {'model_index': [model_index]}

    # Here we ensure that we remove object that are not serializable from the dictionnary (such as Keras optimizer object or kernel regularizer objects)

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
        new_csv = pd.concat([csv_input, pd.DataFrame.from_dict(csv_dict_parameters)], sort=False)
        new_csv.to_csv(benchmark_filename, index=False)
    else:
        pd.DataFrame.from_dict(csv_dict_parameters).to_csv(benchmark_filename, index=False)

    benchmark_filename = os.path.join(all_models_path, 'Benchmark_ANN_performance.csv')

    if os.path.isfile(benchmark_filename):
        csv_input = pd.read_csv(benchmark_filename)
        new_csv = pd.concat([csv_input, pd.DataFrame.from_dict(csv_dict_performance)], sort=False)
        new_csv.to_csv(benchmark_filename, index=False)
    else:
        pd.DataFrame.from_dict(csv_dict_performance).to_csv(benchmark_filename, index=False)

    return

def read_fits(full_path):

    ####role####
    # Read a .fits file and returns the data and keys contained in the latter.

    ####inputs####
    # full_path (string) : Full path of the .fits file to load

    ####outputs####
    # data_keys (list) : The keys corresponding to the data inside the .fits file (eg DES_r, HSC_zphot, ...)
    # data (numpy array) : Object containing the data of the .fits file. The object has the same behavior than

    hdu = fits.open(full_path)
    data = hdu[1].data
    data_keys = hdu[1].columns.names

    nbr_features = len(data_keys)
    nbr_objects = len(data)
    data_array = np.zeros((nbr_objects, nbr_features), dtype=np.float64)

    for i in range(nbr_features):
        data_array[:,i] = data.field(i)

    return data_array, data_keys

def read_fits_as_dict(full_path):

    ####role####
    # Read a .fits file and returns the data and keys contained in the latter.

    ####inputs####
    # path (string) : path of the .fits file to load
    # filename (string) : filename of the .fits file to load

    ####outputs####
    # data_keys (list) : The keys corresponding to the data inside the .fits file (eg DES_r, HSC_zphot, ...)
    # data (data fits object from astropy) : Object containing the data of the .fits file. The object has the same behavior than
    #                                        python dictionnary (eg to access all the DES_r data just type data['DES_r'])

    hdu = fits.open(full_path)
    data = hdu[1].data
    data_keys = hdu[1].columns.names

    return data,

def post_process_color_cut(dataset, dataset_keys, Y_pred):
    Y_pred = np.array(Y_pred)
    unique, counts = np.unique(Y_pred, return_counts=True)

    BG_pred_idx = np.where(Y_pred == 1.0)[0]
    LRG_pred_idx = np.where(Y_pred == 2.0)[0]
    ELG_pred_idx = np.where(Y_pred == 3.0)[0]

    BG_pred_colorcut_mask = compute_BG_color_cut(dataset[BG_pred_idx, :], dataset_keys)
    LRG_pred_colorcut_mask = compute_LRG_color_cut(dataset[LRG_pred_idx, :], dataset_keys)
    ELG_pred_colorcut_mask = compute_ELG_color_cut(dataset[ELG_pred_idx, :], dataset_keys)

    wrong_color_indexes = list(BG_pred_idx[~BG_pred_colorcut_mask]) + list(LRG_pred_idx[~LRG_pred_colorcut_mask]) + list(ELG_pred_idx[~ELG_pred_colorcut_mask])
    for i in wrong_color_indexes:
        Y_pred[i] = 0.0

    return Y_pred

def compute_classnames(classification_problem):

    ####role####
    # Compute classnames from given classification problem. One should note that this function is arbitrary and should be adapted if the user consider other classification poroblems or labels.

    ####inputs####
    # classification_problem (string) : The name of the classification problem considred

    ####outputs####
    # classnames (list of strings) : The class labels for the classification problem considred

    if 'BG_LRG_ELG_QSO_classification' in classification_problem:
        classnames = ['Others', 'BG', 'LRG', 'ELG', 'QSO']

    return classnames

def compute_color_cuts_performances(X_train, zphot_train, X_val, zphot_val, X_test, zphot_test, data_keys_train):

    time_start = time.time()

    color_cut_performances = {
                                'train': {
                                            'BG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   },
                                            'LRG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   },
                                            'ELG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   }
                                         },
                                'val': {
                                            'BG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   },
                                            'LRG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   },
                                            'ELG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   }
                                         },
                                'test': {
                                            'BG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   },
                                            'LRG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   },
                                            'ELG': {
                                                    'recall': 1.0,
                                                    'precision': 0.0,
                                                    'f1-score': 0.0,
                                                   }
                                         },
                                'macro_recall': 0.0,
                                'macro_precision': 0.0,
                                'macro_f1-score': 0.0
                            }

    for key, mags, zphot in zip(['train', 'val', 'test'], [X_train, X_val, X_test], [zphot_train, zphot_val, zphot_test]):

        iscolor_cut = compute_BG_color_cut(mags, data_keys_train)
        iszphot = (
                    (zphot > 0.05) &
                    (zphot < 0.4)
                  )
        nbr_object_colorcut = sum(iscolor_cut)
        nbr_object_of_interest = sum(iscolor_cut & iszphot)
        color_cut_performances[key]['BG']['precision'] = round(nbr_object_of_interest/nbr_object_colorcut, 3)
        color_cut_performances[key]['BG']['f1-score'] = round(2.0*(color_cut_performances[key]['BG']['precision']*color_cut_performances[key]['BG']['recall'])/(color_cut_performances[key]['BG']['precision']+color_cut_performances[key]['BG']['recall']), 3)
        color_cut_performances['macro_recall'] +=  color_cut_performances[key]['BG']['recall']
        color_cut_performances['macro_precision'] += color_cut_performances[key]['BG']['precision']
        color_cut_performances['macro_f1-score'] += color_cut_performances[key]['BG']['f1-score']

        iscolor_cut = compute_LRG_color_cut(mags, data_keys_train)
        iszphot = (
                    (zphot > 0.4) &
                    (zphot < 0.75)
                  )
        nbr_object_colorcut = sum(iscolor_cut)
        nbr_object_of_interest = sum(iscolor_cut & iszphot)
        color_cut_performances[key]['LRG']['precision'] = round(nbr_object_of_interest/nbr_object_colorcut, 3)
        color_cut_performances[key]['LRG']['f1-score'] = round(2.0*(color_cut_performances[key]['LRG']['precision']*color_cut_performances[key]['LRG']['recall'])/(color_cut_performances[key]['LRG']['precision']+color_cut_performances[key]['LRG']['recall']), 3)
        color_cut_performances['macro_recall'] +=  color_cut_performances[key]['LRG']['recall']
        color_cut_performances['macro_precision'] += color_cut_performances[key]['LRG']['precision']
        color_cut_performances['macro_f1-score'] += color_cut_performances[key]['LRG']['f1-score']

        iscolor_cut = compute_ELG_color_cut(mags, data_keys_train)
        iszphot = (
                    (zphot > 0.75) &
                    (zphot < 1.1)
                  )
        nbr_object_colorcut = sum(iscolor_cut)
        nbr_object_of_interest = sum(iscolor_cut & iszphot)
        color_cut_performances[key]['ELG']['precision'] = round(nbr_object_of_interest/nbr_object_colorcut, 3)
        color_cut_performances[key]['ELG']['f1-score'] = round(2.0*(color_cut_performances[key]['ELG']['precision']*color_cut_performances[key]['ELG']['recall'])/(color_cut_performances[key]['ELG']['precision']+color_cut_performances[key]['ELG']['recall']), 3)
        color_cut_performances['macro_recall'] +=  color_cut_performances[key]['ELG']['recall']
        color_cut_performances['macro_precision'] += color_cut_performances[key]['ELG']['precision']
        color_cut_performances['macro_f1-score'] += color_cut_performances[key]['ELG']['f1-score']

    color_cut_performances['macro_recall'] = round(color_cut_performances['macro_recall']/9.0, 3)
    color_cut_performances['macro_precision'] = round(color_cut_performances['macro_precision']/9.0, 3)
    color_cut_performances['macro_f1-score'] = round(color_cut_performances['macro_f1-score']/9.0, 3)

    pprint.pprint(color_cut_performances, width=1)
    print('Evaluation took : ', time.time() - time_start)

    return color_cut_performances

def compute_BG_color_cut(mags, data_keys_train):

    VHS_j_idx = data_keys_train.index('VHS_j')
    VHS_k_idx = data_keys_train.index('VHS_k')
    WISE_w1_idx = data_keys_train.index('WISE_w1')

    x = mags[:, VHS_j_idx] - mags[:, VHS_k_idx]
    y = mags[:, VHS_j_idx] - mags[:, WISE_w1_idx]
    iscolor_cut = (
                    (16<mags[:,VHS_j_idx]) &
                    (mags[:,VHS_j_idx]<18) &
                    (x>0.10) &
                    (x<1.00) &
                    (y>1.6*x-1.6) &
                    (y<1.6*x-0.5) &
                    (y>-0.5*x-1.0) &
                    (y<-0.5*x+0.1)
                  )

    return iscolor_cut

def compute_LRG_color_cut(mags, data_keys_train):

    VHS_j_idx = data_keys_train.index('VHS_j')
    VHS_k_idx = data_keys_train.index('VHS_k')
    WISE_w1_idx = data_keys_train.index('WISE_w1')

    x = mags[:, VHS_j_idx] - mags[:, VHS_k_idx]
    y = mags[:, VHS_j_idx] - mags[:, WISE_w1_idx]
    iscolor_cut = (
                    (18<mags[:,VHS_j_idx]) &
                    (mags[:,VHS_j_idx]<19.5) &
                    (x>0.25) &
                    (x<1.50) &
                    (y<1.50) &
                    (y>1.6*x-1.5) &
                    (y>-0.5*x+0.65)
                  )
    return iscolor_cut

def compute_ELG_color_cut(mags, data_keys_train):

    DES_g_idx = data_keys_train.index('DES_g')
    DES_r_idx = data_keys_train.index('DES_r')
    DES_i_idx = data_keys_train.index('DES_i')

    x = mags[:, DES_g_idx] - mags[:, DES_r_idx]
    y = mags[:, DES_r_idx] - mags[:, DES_i_idx]
    iscolor_cut = (
                    (21<mags[:,DES_g_idx]) &
                    (mags[:,DES_g_idx]<23.2) &
                    (0.5-2.5*x<y) &
                    (y<3.5-2.5*x) &
                    (0.4*x+0.3<y) &
                    (y<0.4*x+0.9)
                  )

    return iscolor_cut

def compute_aucs(Y_pred, Y_val, X_val_id, X_val, X_val_keys, classnames, color_cut_performances, savepath):

    ####role####
    # Compute Area Under Curves given predictions and ground truth

    ####inputs####
    # Y_pred : Predictions of the neural network (one hot encoded format)
    # Y_val : Ground truth corresponding to predictions (one hot encoded format)
    # classnames : The class names (string) corresponding to one hot encoded labels contained in Y_pred and Y_true

    ####outputs####
    # mean_auc_roc : Mean of the area under curve of Receiver Operator Curves (ROC) among classes
    # mean_auc_roc : Mean of the area under curve of Precision Recall (PR) curves among classes

    # First we define the thresholds on the confidence value of the NN predictions for which we want to compute the confusion matrices
    # The number of thresholds used will also define the number of points in the ROC and PR curves

    thresholds = list(np.linspace(0.00, 0.95, 20, endpoint=True))
    thresholds = [ round(elem, 2) for elem in thresholds ]

    # A dictionnary is defined for each metric of interest.
    # terminology : fpr : false positive rate (or fall out)
    #               tpr : true positive rate (or sensitivity)
    #               ppv : positive predicted value (or precision)
    # Each of those dictionnary will store the values of its corresponding metric for each threshold and for each class.
    # To be more specific: - The keys of the dictionnaries are the class numeric labels.
    #                      - To each keys is associated a list corresponding to the metric computed for each threshold in ascending order
    # For more informations regarding those metrics please refer to their in-depth definition at : https://en.wikipedia.org/wiki/Precision_and_recall

    fpr = {}
    tpr = {}
    ppv = {}

    # A list is defined to store the Area Under Curve for each classes.
    # Here we don't nneed a dictionnary as there is only one value of AUC for each class.
    auc_pr = []
    auc_roc = []
    nbr_classes = len(classnames)

    # The matplotlib figure object used to plot confusion matrices are defined here.
    # The reason for this is that confusion matrices for each thresholds are plotted on the same figure and so the figure has to be defined outside of the loop on thresholds value.

    norm_fig = plt.figure(1, figsize=(19.2,12), dpi=100)
    norm_fig.subplots_adjust(left  = 0.1, bottom = 0.05, right = 0.9, top = 1.0, wspace = 0.9, hspace = 0.65)

    fig = plt.figure(2, figsize=(19.2,12), dpi=100)
    fig.subplots_adjust(left  = 0.1, bottom = 0.05, right = 0.9, top = 1.0, wspace = 0.9, hspace = 0.65)

    # In order to compute the different metrics (i.e fpr, tpr, ppv) for each thresholds, one has to compute the confusion matrix for each of those thresholds
    # The confusion matrix is cimputed in compute_conf_matr function

    for idxplot, i in enumerate(thresholds.copy()):
        conf_matr, recall_correction = compute_conf_matr(Y_val, Y_pred, X_val_id, X_val, X_val_keys, i, savepath)
        # We ensure that the confusion matric has the same number of rows than the number of classes. This is due to the fact that if given a threshold there is no predictions falling in one of the class,
        # the confusion matrix shape will not match the number of classes and this can give error in computations of metrics of interests
        if conf_matr.shape[0] == nbr_classes:
            fpr, tpr, ppv = compute_roc_pr_curve(conf_matr, fpr, tpr, ppv, nbr_classes, recall_correction)
            plot_confusion_matrix(conf_matr, classnames, idxplot, i, tpr, ppv, savepath, True, 1)
            plot_confusion_matrix(conf_matr, classnames, idxplot, i, tpr, ppv, savepath, False, 2)
        else:
            thresholds.remove(i)

    # Depending on the curve that is plotted, either fpr or tpr are used on the x-axis
    # We have to ensure that the set of points given to matplotlib are in ascending order with respect to the x-axis
    # Thus, the dictionnary are reordered for each of the computed curve (i.e ROC and PR)

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

    plt.close(1)
    plt.close(2)
    plot_roc_curve(fpr_roc, tpr_roc, ppv, thresholds_roc, nbr_classes, classnames, savepath, auc_roc)

    for k in range(nbr_classes):
        idx_redordering =  np.argsort(tpr[k])
        tpr_pr[k] = [tpr[k][h] for h in idx_redordering]
        ppv_pr[k] = [ppv[k][h] for h in idx_redordering]
        thresholds_pr = [thresholds[h] for h in idx_redordering]
        auc_pr.append(np.trapz(y=([1] + ppv_pr[k] + [0]), x=([0] + tpr_pr[k] + [1])))

    plot_pr_curve(fpr, tpr_pr, ppv_pr, thresholds_pr, nbr_classes, classnames, color_cut_performances, savepath, auc_pr)

    mean_auc_pr = sum(auc_pr)/len(auc_pr)
    mean_auc_roc = sum(auc_roc)/len(auc_roc)

    return mean_auc_roc, mean_auc_pr

def compute_conf_matr(Y_val, Y_pred, X_val_id, X_val, X_val_keys, threshold, savepath):

    ####role####
    # Compute Confusion Matrix given predictions and ground truth

    ####inputs####
    # Y_pred : Predictions of the neural network (one hot encoded format)
    # Y_val : Ground truth corresponding to predictions (one hot encoded format)
    # theshold : The threshold for which we want to compute the confusion matrix

    ####outputs####
    # cm : The confusion matrix computed for the given threshold
    # recall_correction : The recall correction that will be used to compute metrics in parent function

    # First we count the number of object in each class and store this information in a dictionnary
    unique, counts = np.unique(np.argmax(Y_val, axis=-1), return_counts=True)
    class_count_dict = dict(zip(unique, counts))
    nbr_classes = len(list(class_count_dict.keys()))

    # We define a two list that will contain the numeric laebls for predictions and ground truth computed from the one hot encoded labels given as inputs
    Y_pred_non_category = []
    Y_val_non_category = []
    DES_id = []
    X_val_idx_kept = []
    title = 'Threshold = ' + str(threshold)

    ####Note on thresholds and recall####
    # It is possible that there is no label that have a confidence score that respect the threshold condition.
    # In this case, the prediction is discarded. However, in order to have a comparable recall among threshold, one should compute the impact of discarding those predictions on the recall.
    # To do so, a dictionnary is defned in order to keep count of the number of discarded predictions in each class.
    # Then this dictionnary is used to bring a correction factor on the recall for each threshold.

    discard_count_dict = {}
    for i in list(class_count_dict.keys()):
        discard_count_dict[i] = 0
    recall_correction = {'macro': 0, 'micro': 0}

    # We transform the one hot encoded labels to numeric laebsl using argmax

    for idxj, j in enumerate(Y_pred):
        if np.amax(j) >= threshold:
            Y_val_non_category.append(np.argmax(Y_val[idxj]))
            Y_pred_non_category.append(np.argmax(Y_pred[idxj]))
            DES_id.append(X_val_id[idxj])
            X_val_idx_kept.append(idxj)
        else:
            discard_count_dict[np.argmax(Y_val[idxj])] += 1

    for i in list(class_count_dict.keys()):
        recall_correction[i] = 1 - discard_count_dict[i]/class_count_dict[i]
        recall_correction['macro'] += recall_correction[i]

    recall_correction['macro'] = recall_correction['macro']/nbr_classes
    recall_correction['micro'] = 1 - sum(list(discard_count_dict.values()))/(sum(list(class_count_dict.values())))

    if 'post_processed' in savepath:
        Y_pred_non_category = post_process_color_cut(X_val[X_val_idx_kept, :], X_val_keys, Y_pred_non_category)

    if savepath is not None:
        prediction_report = {'DES_id': DES_id, 'Y_true': Y_val_non_category, 'Y_pred': Y_pred_non_category}
        pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(savepath, 'Predictions_' + str(threshold) + '.csv'), index=False)

    # The confusion matrix is computed using sklearn function (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

    cm = confusion_matrix(Y_val_non_category, Y_pred_non_category)

    return cm, recall_correction

def compute_roc_pr_curve(confusion_matrix, fpr, tpr, ppv, n_classes, recall_correction):

    ####role####
    # Compute Metrics of interest given a confusion matrix

    ####inputs####
    # confusion_matrix (2D numpy array with shape n_classes*n_classes): The confusion matrix computed from sklearn
    # fpr (dictionnray) : False Positive Rate dictionnary from parent function
    # tpr (dictionnray) : True Positive Rate dictionnary from parent function
    # ppv (dictionnray) : Positive Predicted Value dictionnary from parent function
    # n_classes (int) : The number of classes
    # recall_correction (dictionnray) : The recall correction to use in recall computation for each class

    ####outputs####
    # fpr (dictionnray) : False Positive Rate dictionnary computed for the confusion matrix
    # tpr (dictionnray) : True Positive Rate dictionnary from parent function
    # ppv (dictionnray) : Positive Predicted Value dictionnary from parent function

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

        if (i in list(tpr.keys())) and (i in list(fpr.keys())) and (i in list(ppv.keys())):
            if (TP+FN) == 0.0:
                tpr[i].append(0.0)
            else:
                tpr[i].append(TP/(TP+FN)*recall_correction[i])
            if (FP+TN) == 0.0:
                fpr[i].append(0.0)
            else:
                fpr[i].append(FP/(FP+TN))
            if (FP+TP) == 0.0:
                ppv[i].append(0.0)
            else:
                ppv[i].append(TP/(FP+TP))
        else:
            if (TP+FN) == 0.0:
                tpr[i] = [0.0]
            else:
                tpr[i] = [TP/(TP+FN)*recall_correction[i]]
            if (FP+TN) == 0.0:
                fpr[i] = [0.0]
            else:
                fpr[i] = [FP/(FP+TN)]
            if (FP+TP) == 0.0:
                ppv[i] = [0.0]
            else:
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
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)
        plt.grid(linestyle='-', linewidth='0.5', color='grey')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Area Under Curve of class {} : {:0.2f}'.format(classnames[i], auc_roc[i]))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(savepath, 'roc_' + classnames[i] + '.png'))
        plt.close()

    return

def plot_pr_curve(fpr, tpr, ppv, thresholds, n_classes, classnames, color_cut_performances, savepath, auc_pr):

    colors = ['blue', 'brown', 'cyan', 'red', 'yellow', 'magenta', 'lime', 'orange', 'purple', 'lightgray', 'gold', 'darkblue', 'tan', 'olive', 'turquoise', 'pink', 'darkgreen', 'gray', 'lightsalmon', 'black']
    levels_labels = {}
    thresholds_copy = thresholds.copy()
    colors_copy = colors.copy()

    for i in range(n_classes):
        fig = plt.figure(figsize=(19.2,10.8), dpi=100)
        for idx in range(len(thresholds_copy)):
            plt.scatter(tpr[i][idx], ppv[i][idx], color=colors_copy[idx],
                     label='Threshold = {0})'
                     ''.format(thresholds_copy[idx]))
        plt.step([0] + tpr[i] + [1],[0] + ppv[i] + [0], linestyle=':' ,lw=1, color='black')
        plt.plot([0, 1], [0, 0], 'k--', lw=2)
        ax = fig.axes[0]

        recall_f1 = np.arange(0.01,1.01,0.01)
        precision_f1 = np.arange(0.01,1.01,0.01)
        levels = list(np.arange(0.1,1.1,0.1))
        if classnames[i] in ['BG', 'LRG', 'ELG']:
            if color_cut_performances[classnames[i]]['f1-score'] not in levels:
                levels.append(color_cut_performances[classnames[i]]['f1-score'])
                levels = sorted(levels)
            red = Color("red")
            colors_f1 = list(red.range_to(Color("green"),len(levels)))
            idx_color_cut = levels.index(color_cut_performances[classnames[i]]['f1-score'])
            colors_f1[idx_color_cut] = Color("cyan")
            for j in levels:
                if j == color_cut_performances[classnames[i]]['f1-score']:
                    levels_labels[j] = 'Cut:f1-score=' + "%.3f" % round(j,3)
                else:
                    levels_labels[j] = 'f1-score='+ "%.3f" % round(j,3)
        else:
            red = Color("red")
            colors_f1 = list(red.range_to(Color("green"),len(levels)))
            for j in levels:
                levels_labels[j] = 'f1-score = ' + "%.3f" % round(j,3)
        colors_f1 = [c.hex_l for c in colors_f1]
        X_f1, Y_f1 = np.meshgrid(recall_f1, precision_f1)
        f1_score = 2*(X_f1*Y_f1)/(X_f1 + Y_f1)
        CS = ax.contour(X_f1, Y_f1, f1_score, levels=levels, colors=colors_f1)
        ax.clabel(CS, CS.levels, fmt=levels_labels, inline=1, fontsize=10)

        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)
        plt.grid(linestyle='-', linewidth='0.5', color='grey')
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
    return dict_list

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

def load_dataset(dataset_path, model_path, classification_problem, zphot_safe_threshold, train_test_val_split, cv_fold_nbr, normalization, data_imputation, random_seed):

    training_dataset_filename = classification_problem + '_' + str(zphot_safe_threshold) + '-zsafe' + '_train_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '_' + str(cv_fold_nbr) + '.fits'
    validation_dataset_filename = classification_problem + '_' + str(zphot_safe_threshold) + '-zsafe' + '_val_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '_' + str(cv_fold_nbr) + '.fits'
    testing_dataset_filename = classification_problem + '_' + str(zphot_safe_threshold) + '-zsafe' + '_test_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '.fits'

    # load  dataset
    training_dataset, data_keys = read_fits(os.path.join(dataset_path, training_dataset_filename))
    np.random.shuffle(training_dataset)
    validation_dataset, _ = read_fits(os.path.join(dataset_path, validation_dataset_filename))
    np.random.shuffle(validation_dataset)
    testing_dataset, _ = read_fits(os.path.join(dataset_path, testing_dataset_filename))
    np.random.shuffle(testing_dataset)

    # split into input (X) and output (Y) variables
    DES_id_train = training_dataset[:,0]
    zphot_train = training_dataset[:,1]
    sample_weights_train = training_dataset[:,2]
    data_keys_train = data_keys[3:-1]
    X_train = training_dataset[:,3:-1]
    Y_train = training_dataset[:,-1]

    print(data_keys_train)
    print(X_train.shape[1])

    save_training_data(X_train, data_keys_train, model_path)

    # split into input (X) and output (Y) variables
    DES_id_val = validation_dataset[:,0]
    zphot_val = validation_dataset[:,1]
    sample_weights_val = validation_dataset[:,2]
    X_val = validation_dataset[:,3:-1]
    Y_val = validation_dataset[:,-1]

    # split into input (X) and output (Y) variables
    DES_id_test = testing_dataset[:,0]
    zphot_test = testing_dataset[:,1]
    sample_weights_test = testing_dataset[:,2]
    X_test = testing_dataset[:,3:-1]
    Y_test = testing_dataset[:,-1]

    return DES_id_train, X_train, Y_train, zphot_train, sample_weights_train, DES_id_val, X_val, Y_val, zphot_val, sample_weights_val, DES_id_test, X_test, Y_test, zphot_test, sample_weights_test, data_keys, data_keys_train

def load_full_dataset(dataset_path, classification_problem, data_imputation):
    dataset_filename = classification_problem + '.fits'
    # load  dataset
    full_dataset, dataset_keys = read_fits(os.path.join(dataset_path, dataset_filename))
    np.random.shuffle(full_dataset)

    X = full_dataset[:,0:-1]
    Y = full_dataset[:,-1]

    nbr_features = X.shape[1]

    for i in range(nbr_features):
        nan_mask = np.isnan(X[:,i])
        not_nan_mask = [not j for j in nan_mask]
        idx = np.where(not_nan_mask)[0]
        mean = np.mean(X[idx,i])
        std_var = np.std(X[idx, i])
        X[:,i] = (X[:,i] - mean)/std_var
        if data_imputation == 'max':
            nan_idx = np.where(nan_mask)[0]
            notnan_idx = np.where(not_nan_mask)[0]
            X[nan_idx,i] = np.amax(X[notnan_idx,i])
        else:
            nan_mask = np.isnan(X[:,i])
            nan_idx = np.where(nan_mask)[0]
            X[idx,i] = data_imputation

    return X, Y, dataset_keys

# def compute_weights(Y):
#
#     class_weights = compute_class_weights(Y)
#     sample_weights = compute_sample_weights(Y, class_weights)
#
#     return sample_weights
#
# def compute_sample_weights(Y, class_weights):
#
#     unique, counts = np.unique(Y, return_counts=True)
#     nbr_objects = Y.shape[0]
#     class_count_dict = dict(zip(unique, counts))
#     all_labels = list(class_count_dict.keys())
#     sample_weights = []
#
#     for i in Y:
#         for j in all_labels:
#             if i == j:
#                 sample_weights.append(class_weights[j])
#
#     return np.array(sample_weights)
#
# def compute_class_weights(Y):
#     unique, counts = np.unique(Y, return_counts=True)
#     nbr_objects = Y.shape[0]
#     class_count_dict = dict(zip(unique, counts))
#     labels = list(class_count_dict.keys())
#     nbr_labels = len(labels)
#     class_weights = {}
#     for idx, i in enumerate(labels):
#         class_weights[i] = nbr_objects/(class_count_dict[i])
#
#
#     return class_weights

def one_hot_encode(Y_train, Y_val, Y_test):

    Y_train = np.array(Y_train, dtype='int')
    Y_train = Y_train.reshape(len(Y_train), 1)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    Y_train = onehot_encoder.fit_transform(Y_train)

    Y_val = np.array(Y_val, dtype='int')
    Y_val = Y_val.reshape(len(Y_val), 1)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    Y_val = onehot_encoder.fit_transform(Y_val)

    Y_test = np.array(Y_test, dtype='int')
    Y_test = Y_test.reshape(len(Y_test), 1)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    Y_test = onehot_encoder.fit_transform(Y_test)

    return Y_train, Y_val, Y_test

def apply_normalization(dataset, normalization, normalizer):

    if normalization == 'cr':
        dataset_norm = apply_reduce_center(dataset, normalizer)
        return dataset_norm
    elif normalization == 'quant':
        dataset_norm  = apply_quantile_transform(dataset, normalizer)
        return dataset_norm

def unapply_reduce_center(dataset, mean_std):

    dataset_uncr = dataset.copy()

    for i in range(mean_std.shape[0]):
        dataset_uncr[:,i] = (dataset[:,i]*mean_std[i, 1]) + mean_std[i, 0]

    return dataset_uncr

def apply_reduce_center(dataset, mean_std):

    dataset_cr = dataset.copy()

    for i in range(mean_std.shape[0]):
        dataset_cr[:,i] = (dataset_cr[:,i] - mean_std[i, 0])/mean_std[i, 1]

    return dataset_cr

def apply_quantile_transform(dataset, quantile_transformer):

    dataset_quant = dataset.copy()
    dataset_quant = quantile_transformer.transform(dataset_quant)

    return dataset_quant

def fill_empty_entries(dataset, data_imputation_values):

    nbr_features = dataset.shape[1]
    dataset_filled = dataset.copy()

    for i in range(nbr_features):
        nan_mask = np.isnan(dataset[:,i])
        nan_idx = np.where(nan_mask)[0]
        dataset_filled[nan_idx,i] = data_imputation_values[i]

    return dataset_filled

def normalize_dataset(X_train, X_val, X_test, normalization, random_seed):

    if normalization == 'cr':
        X_train_norm, X_val_norm, X_test_norm, mean_std = reduce_and_center_dataset(X_train, X_val, X_test)
        return X_train_norm, X_val_norm, X_test_norm, mean_std
    elif normalization == 'quant':
        X_train_norm, X_val_norm, X_test_norm = quantile_transform_dataset(X_train, X_val, X_test, random_seed)
        return X_train_norm, X_val_norm, X_test_norm, quantile_transformer

def reduce_and_center_dataset(X_train, X_val, X_test):

    nbr_object = X_train.shape[0]
    nbr_features = X_train.shape[1]
    mean_std = np.zeros((nbr_features, 2))
    X_train_cr = X_train.copy()
    X_val_cr = X_val.copy()
    X_test_cr = X_test.copy()

    for i in range(nbr_features):
        nan_mask = np.isnan(X_train[:,i])
        idx = np.where(~nan_mask)[0]
        mean_std[i, 0] = np.mean(X_train[idx,i])
        mean_std[i, 1] = np.std(X_train[idx, i])
        print(i, mean_std[i, 0])
        print(i, mean_std[i, 1])
        X_train_cr[:,i] = (X_train_cr[:,i] - mean_std[i, 0])/mean_std[i, 1]
        X_val_cr[:,i] = (X_val_cr[:,i] - mean_std[i, 0])/mean_std[i, 1]
        X_test_cr[:,i] = (X_test_cr[:,i] - mean_std[i, 0])/mean_std[i, 1]

    return X_train_cr, X_val_cr, X_test_cr, mean_std

def quantile_transform_dataset(X_train, X_val, X_test, random_seed):

    X_train_quant = X_train.copy()
    X_val_quant = X_val.copy()
    X_test_quant = X_test.copy()

    quantile_transformer = QuantileTransformer(random_state=random_seed)
    X_train_quant = quantile_transformer.fit_transform(X_train_quant)
    X_val_quant = quantile_transformer.transform(X_val_quant)
    X_test_quant = quantile_transformer.transform(X_test_quant)

    return X_train_quant, X_val_quant, X_test_quant, quantile_transformer

def fill_empty_entries_dataset(X_train, X_val, X_test, data_imputation, constraints):

    nbr_object = X_train.shape[0]
    nbr_features = X_train.shape[1]
    X_train_filled = X_train.copy()
    X_val_filled = X_val.copy()
    X_test_filled = X_test.copy()

    if data_imputation == 'max':

        max_features = np.zeros(nbr_features)

        if 'colors' in constraints:

            for i in range((nbr_features-1),-1,-1):
                if i ==0:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,6]) - np.nanmax(X_train[:,7]), np.nanmax(X_val[:,6] - X_val[:,7]), np.nanmax(X_test[:,6] - X_test[:,7])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i ==1:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,7]) - np.nanmax(X_train[:,8]), np.nanmax(X_val[:,7] - X_val[:,8]), np.nanmax(X_test[:,7] - X_test[:,8])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i==2:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,8]) - np.nanmax(X_train[:,9]), np.nanmax(X_val[:,8] - X_val[:,9]), np.nanmax(X_test[:,8] - X_test[:,9])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i==3:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,9]) - np.nanmax(X_train[:,10]), np.nanmax(X_val[:,9] - X_val[:,10]), np.nanmax(X_test[:,9] - X_test[:,10])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i==4:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,11]) - np.nanmax(X_train[:,12]), np.nanmax(X_val[:,11] - X_val[:,12]), np.nanmax(X_test[:,11] - X_test[:,12])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i==5:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,11]) - np.nanmax(X_train[:,13]), np.nanmax(X_val[:,11] - X_val[:,13]), np.nanmax(X_test[:,11] - X_test[:,13])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                else:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,i]), np.nanmax(X_val[:,i]), np.nanmax(X_test[:,i])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]

        else:

            for i in range((nbr_features-1),-1,-1):

                max_features[i] = math.ceil(max(np.nanmax(X_train[:,i]), np.nanmax(X_val[:,i]), np.nanmax(X_test[:,i])))
                nan_mask = np.isnan(X_train[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_train_filled[nan_idx,i] = max_features[i]
                nan_mask = np.isnan(X_val[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_val_filled[nan_idx,i] = max_features[i]
                nan_mask = np.isnan(X_test[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_test_filled[nan_idx,i] = max_features[i]

        data_imputation_values = max_features

    elif data_imputation == 'mean':

        mean_features = np.zeros(nbr_features)

        if 'colors' in constraints:

            for i in range((nbr_features-1),-1,-1):
                if i ==0:
                    mean_features[i] = math.ceil(mean(np.nanmean(X_train[:,6]) - np.nanmean(X_train[:,7]), np.nanmean(X_val[:,6] - X_val[:,7]), np.nanmean(X_test[:,6] - X_test[:,7])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i ==1:
                    mean_features[i] = math.ceil(mean(np.nanmean(X_train[:,7]) - np.nanmean(X_train[:,8]), np.nanmean(X_val[:,7] - X_val[:,8]), np.nanmean(X_test[:,7] - X_test[:,8])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i==2:
                    mean_features[i] = math.ceil(mean(np.nanmean(X_train[:,8]) - np.nanmean(X_train[:,9]), np.nanmean(X_val[:,8] - X_val[:,9]), np.nanmean(X_test[:,8] - X_test[:,9])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i==3:
                    mean_features[i] = math.ceil(mean(np.nanmean(X_train[:,9]) - np.nanmean(X_train[:,10]), np.nanmean(X_val[:,9] - X_val[:,10]), np.nanmean(X_test[:,9] - X_test[:,10])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i==4:
                    mean_features[i] = math.ceil(mean(np.nanmean(X_train[:,11]) - np.nanmean(X_train[:,12]), np.nanmean(X_val[:,11] - X_val[:,12]), np.nanmean(X_test[:,11] - X_test[:,12])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i==5:
                    mean_features[i] = math.ceil(mean(np.nanmean(X_train[:,11]) - np.nanmean(X_train[:,13]), np.nanmean(X_val[:,11] - X_val[:,13]), np.nanmean(X_test[:,11] - X_test[:,13])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                else:
                    mean_features[i] = math.ceil(mean(np.nanmean(X_train[:,i]), np.nanmean(X_val[:,i]), np.nanmean(X_test[:,i])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]

        else:

            for i in range((nbr_features-1),-1,-1):

                mean_features[i] = math.ceil(mean(np.nanmean(X_train[:,i]), np.nanmean(X_val[:,i]), np.nanmean(X_test[:,i])))
                nan_mask = np.isnan(X_train[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_train_filled[nan_idx,i] = mean_features[i]
                nan_mask = np.isnan(X_val[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_val_filled[nan_idx,i] = mean_features[i]
                nan_mask = np.isnan(X_test[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_test_filled[nan_idx,i] = mean_features[i]

        data_imputation_values = mean_features

    elif data_imputation == 'maxmean':

        max_features = np.zeros(nbr_features)
        mean_features = np.zeros(nbr_features)
        maxmean_features = np.zeros(nbr_features)

        for i in range((nbr_features-1),-1,-1):

            max_features[i] = max(np.nanmax(X_train[:,i]), np.nanmax(X_val[:,i]), np.nanmax(X_test[:,i]))
            mean_features[i] = mean(np.nanmean(X_train[:,i]), np.nanmean(X_val[:,i]), np.nanmean(X_test[:,i]))
            maxmean_features[i] = (max_features[i] + mean_features[i])/2.0
            nan_mask = np.isnan(X_train[:,i])
            nan_idx = np.where(nan_mask)[0]
            X_train_filled[nan_idx,i] = maxmean_features[i]
            nan_mask = np.isnan(X_val[:,i])
            nan_idx = np.where(nan_mask)[0]
            X_val_filled[nan_idx,i] = maxmean_features[i]
            nan_mask = np.isnan(X_test[:,i])
            nan_idx = np.where(nan_mask)[0]
            X_test_filled[nan_idx,i] = maxmean_features[i]

        data_imputation_values = maxmean_features

    else:

        for i in range(nbr_features):
            nan_mask = np.isnan(X_train[:,i])
            idx = np.where(nan_mask)[0]
            X_train_filled[idx,i] = data_imputation
            nan_mask = np.isnan(X_val[:,i])
            idx = np.where(nan_mask)[0]
            X_val_filled[idx,i] = data_imputation
            nan_mask = np.isnan(X_test[:,i])
            idx = np.where(nan_mask)[0]
            X_test_filled[idx,i] = data_imputation

        data_imputation_values = np.full(nbr_features, data_imputation)



    return X_train_filled, X_val_filled, X_test_filled, data_imputation_values

def save_training_data(X_train, data_keys_train, model_path):

    fits_column_train = []
    nbr_features_train = X_train.shape[1]
    for g in range(nbr_features_train):
        fits_column_train.append(fits.Column(name=data_keys_train[g], array=X_train[:,g], format='E'))
    training_data = fits.BinTableHDU.from_columns(fits_column_train)
    try:
        training_data.writeto(os.path.join(model_path, 'training_data.fits'))
    except OSError as e:
        print(e)

    return
