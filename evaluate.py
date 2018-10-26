import __init__

import os
import csv
import numpy as np
import json
from scipy import interp
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
import itertools
from itertools import cycle
import pandas as pd
import csv
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from colour import Color

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

def get_model_and_weights(model_path, dataset_suffix):

    # load json and create model
    json_file = open(os.path.join(model_path, dataset_suffix + '.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(model_path, dataset_suffix + '_final.h5'))

    with open(os.path.join(model_path, 'ann_parameters.json')) as f:
        ann_parameters = json.load(f)

    # evaluate loaded model on test data
    loaded_model.compile(loss=ann_parameters['loss_function'], optimizer=Adam(lr=ann_parameters['learning_rate']), metrics=[])

    return loaded_model

def plot_confusion_matrix(cm, title='Confusion matrix', target_names=None, idx_plot=1, cmap=None, normalize=False, savepath=None, dataset_suffix=None):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.subplot(4, 5, idx_plot + 1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

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
                     color="red" if cm[i, j] > thresh else "black", fontsize=6)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black", fontsize=6)


    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    if normalize:
        plt.savefig(os.path.join(savepath, dataset_suffix + '_cm_norm.png'))
    else:
        plt.savefig(os.path.join(savepath, dataset_suffix + '_cm.png'))

def compute_confusion_matrix(model, Y_test, Y_pred, X_test_id, threshold=None, classnames=None, idx_plot=1, title='Confusion matrix', cmap=None, normalize=False, savepath=None, dataset_suffix=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    n_classes = len(classnames)

    if threshold is None:
        y_test_non_category.append(np.argmax(Y_test[idxj]))
        y_pred_non_category.append(np.argmax(Y_pred[idxj]))
        title = 'Threshold = ' + 'argmax'
        cm = confusion_matrix(y_test_non_category, y_pred_non_category)
    else:
        y_pred_non_category = []
        y_test_non_category = []
        DES_id = []
        title = 'Threshold = ' + str(threshold)
        for idxj, j in enumerate(Y_pred):
            if np.amax(j) >= threshold:
                DES_id.append(X_test_id[idxj])
                y_test_non_category.append(np.argmax(Y_test[idxj]))
                y_pred_non_category.append(np.argmax(Y_pred[idxj]))
        cm = confusion_matrix(y_test_non_category, y_pred_non_category)


    prediction_report = {'DES_id': DES_id, 'Y_true': y_test_non_category, 'Y_pred': y_pred_non_category}
    prediction_report_path, _ = os.path.split(savepath)
    pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(prediction_report_path, 'Predictions_' + str(threshold) + '.csv'), index=False)

    if cm.shape[0] == n_classes:
        plot_confusion_matrix(cm, title, classnames, idx_plot, savepath=savepath, normalize=False, dataset_suffix=dataset_suffix)
        # plot_confusion_matrix(cm, title, classnames, idx_plot, savepath=savepath, normalize=True)
    return cm, n_classes

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

def plot_roc_curve(fpr, tpr, ppv, thresholds, n_classes, classnames, savepath, dataset_suffix):

    colors = ['blue', 'brown', 'cyan', 'red', 'yellow', 'magenta', 'lime', 'orange', 'purple', 'lightgray', 'gold', 'darkblue', 'tan', 'olive', 'turquoise', 'pink', 'darkgreen', 'gray', 'lightsalmon']
    # cyan = Color("blue")
    # colors = list(cyan.range_to(Color("red"),len(thresholds)))
    # colors = [i.hex_l for i in colors]
    roc_auc = {}
    thresholds_copy = thresholds.copy()
    colors_copy = colors.copy()
    for i in range(n_classes):
        idx_redordering = np.argsort(fpr[i])
        fpr[i] = [fpr[i][h] for h in idx_redordering]
        tpr[i] = [tpr[i][h] for h in idx_redordering]
        thresholds_copy = [thresholds[h] for h in idx_redordering]
        colors_copy = [colors[h] for h in idx_redordering]
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(19.2,10.8), dpi=100)
        for idxj, j in enumerate(thresholds_copy):
            plt.scatter(fpr[i][idxj], tpr[i][idxj], color=colors_copy[idxj],
                     label='Threshold = {0})'
                     ''.format(thresholds_copy[idxj]))
        # plt.step([0] + fpr[i] + [1], [0] + tpr[i] + [1], lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Area Under Curve of class {0}'
                  ''.format(classnames[i]))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(savepath, dataset_suffix + '_roc_' + classnames[i] + '.png'))
        plt.close()

    roc_auc = {}
    thresholds_copy = thresholds.copy()
    colors_copy = colors.copy()

    for i in range(n_classes):
        idx_redordering = np.argsort(fpr[i])
        ppv[i] = [ppv[i][h] for h in idx_redordering]
        tpr[i] = [tpr[i][h] for h in idx_redordering]
        thresholds_copy = [thresholds[h] for h in idx_redordering]
        colors_copy = [colors[h] for h in idx_redordering]

        plt.figure(figsize=(19.2,10.8), dpi=100)
        for idxj, j in enumerate(thresholds_copy):
            plt.scatter(tpr[i][idxj], ppv[i][idxj], color=colors_copy[idxj],
                     label='Threshold = {0})'
                     ''.format(thresholds_copy[idxj]))
        # plt.step([0] + fpr[i] + [1], [0] + tpr[i] + [1], lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(' Precision-Recall of class {0}'
                  ''.format(classnames[i]))
        plt.legend(loc="best")
        plt.savefig(os.path.join(savepath, dataset_suffix + '_pre_recall_' + classnames[i] + '.png'))
        plt.close()

    # # Zoom in view of the upper left corner.
    # plt.figure(2)
    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1)
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

def evaluate_run(model_index):

    seed = 7
    np.random.seed(seed)
    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)

    df = pd.read_csv('ANN_Benchmark.csv')
    run_dict = df.to_dict()
    print(model_index)
    for i in list(run_dict['model_index'].keys()):
        if run_dict['model_index'][i] == model_index:
            model_csv_idx = i
    tmp_class_fraction = run_dict['class_fraction'][model_csv_idx].replace('[', '')
    tmp_class_fraction = tmp_class_fraction.replace(']', '')
    tmp_class_fraction = tmp_class_fraction.replace(' ', '')
    tmp_class_fraction = tmp_class_fraction.split(',')

    run_dict['class_fraction'][model_csv_idx] = []
    for i in tmp_class_fraction:
        run_dict['class_fraction'][model_csv_idx].append(float(i))

    dataset_suffix = str(run_dict['train_split'][model_csv_idx]) + '_' + str(run_dict['test_split'][model_csv_idx]) + '_' + str(run_dict['fold_nbr'][model_csv_idx])
    dataset_name = run_dict['classification_problem'][model_csv_idx] + '_test_' + dataset_suffix
    if run_dict['class_fraction'][model_csv_idx][0] > 0.0:
        others_flag = 'all'
    else:
        others_flag = 'no'

    thresholds = list(np.linspace(0.05, 0.95, 19, endpoint=True))
    thresholds = [ round(elem, 2) for elem in thresholds ]
    fpr = {}
    tpr = {}
    ppv = {}

    dataset_path = os.path.join(script_dir, 'datasets', run_dict['catalog'][model_csv_idx], others_flag + '-others_' + run_dict['constraints'][model_csv_idx] + '-constraints', dataset_name + '.fits')
    model_path = os.path.join(script_dir, 'model', str(model_index))

    model = get_model_and_weights(model_path, dataset_suffix)
    classnames = compute_classnames(run_dict['classification_problem'][model_csv_idx], others_flag)
    save_path = os.path.join(model_path, 'figures')

    test_dataset = read_fits(dataset_path)
    np.random.shuffle(test_dataset)

    # split into input (X) and output (Y) variables
    X_test = test_dataset[:, :-1]
    Y_test = test_dataset[:, -1]

    X_test_id = X_test[:, 0]
    X_test = X_test[:, 1:]

    input_dimensions = X_test.shape[1]

    # One hot encoding of labels contained in Y

    Y_test = np.array(Y_test, dtype='int')
    Y_test = Y_test.reshape(len(Y_test), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_test = onehot_encoder.fit_transform(Y_test)
    Y_pred = model.predict(X_test)

    plt.figure(figsize=(19.2,10.8), dpi=100)
    plt.subplots_adjust(left  = 0.1, bottom = 0.04, right = 0.9, top = 0.98, wspace = 0.9, hspace = 0.65)

    for idxplot, i in enumerate(thresholds.copy()):
        conf_matr, nbr_classes = compute_confusion_matrix(model, Y_test, Y_pred, X_test_id, threshold=i, classnames=classnames, idx_plot=idxplot, savepath=save_path, dataset_suffix=dataset_suffix)
        if conf_matr.shape[0] == nbr_classes:
            fpr, tpr, ppv = compute_roc_curve(conf_matr, fpr, tpr, ppv, nbr_classes)
        else:
            thresholds.remove(i)
    plt.close()

    plot_roc_curve(fpr, tpr, ppv, thresholds, nbr_classes, classnames, save_path, dataset_suffix=dataset_suffix)

    return

def evaluate_double_run(first_model_index, second_model_index):

    seed = 7
    np.random.seed(seed)
    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)

    dataset_suffix = '80_20_1'

    model_path = os.path.join(script_dir, 'model', str(first_model_index))

    model_1 = get_model_and_weights(model_path, dataset_suffix)

    thresholds = list(np.linspace(0.05, 0.95, 19, endpoint=True))
    thresholds = [ round(elem, 2) for elem in thresholds ]
    fpr = {}
    tpr = {}
    ppv = {}

    dataset_path = os.path.join(script_dir, 'datasets', '4MOST.CatForGregoire.11Oct2018.zphot', 'all' + '-others_' + 'no' + '-constraints', 'BG_ELG_LRG_QSO_classification_test_80_20_1' + '.fits')
    model_path = os.path.join(script_dir, 'model', str(second_model_index))

    model = get_model_and_weights(model_path, dataset_suffix)
    classnames = compute_classnames('BG_ELG_LRG_QSO_classification', 'all')
    save_path = os.path.join(model_path, 'figures')

    test_dataset = read_fits(dataset_path)
    np.random.shuffle(test_dataset)

    # split into input (X) and output (Y) variables
    X_test = test_dataset[:, :-1]
    Y_test = test_dataset[:, -1]

    X_test_id = X_test[:, 0]
    X_test = X_test[:, 1:]

    input_dimensions = X_test.shape[1]

    # One hot encoding of labels contained in Y

    Y_test = np.array(Y_test, dtype='int')
    Y_test = Y_test.reshape(len(Y_test), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_test = onehot_encoder.fit_transform(Y_test)
    Y_pred_2 = model.predict(X_test)
    Y_pred_1 = model_1.predict(X_test)
    Y_pred = np.zeros((Y_pred_2.shape[0],5))
    for idx, i in enumerate(Y_pred_1):
        Y_pred[idx, :] = np.concatenate((Y_pred_1[idx, 0], Y_pred_1[idx, 1]*Y_pred_2[idx, :]), axis=None)

    plt.figure(figsize=(19.2,10.8), dpi=100)
    plt.subplots_adjust(left  = 0.1, bottom = 0.04, right = 0.9, top = 0.98, wspace = 0.9, hspace = 0.65)

    for idxplot, i in enumerate(thresholds.copy()):
        conf_matr, nbr_classes = compute_confusion_matrix(model, Y_test, Y_pred, X_test_id, threshold=i, classnames=classnames, idx_plot=idxplot, savepath=save_path, dataset_suffix=dataset_suffix)
        if conf_matr.shape[0] == nbr_classes:
            fpr, tpr, ppv = compute_roc_curve(conf_matr, fpr, tpr, ppv, nbr_classes)
        else:
            thresholds.remove(i)
    plt.close()

    plot_roc_curve(fpr, tpr, ppv, thresholds, nbr_classes, classnames, save_path, dataset_suffix=dataset_suffix)

    return
# evaluate_double_run(47, 46)
evaluate_run(63)
# for i in range(43, 48,1):
#     evaluate_run(i)
