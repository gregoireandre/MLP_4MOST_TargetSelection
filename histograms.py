#We want blue -> true positive, red -> false positive, green -> false negative

from astropy.io import fits
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def extract_keys_from_pred(dicti, dict_key, pred_type):
    print(dicti['pred'])
    output = []
    for i in range(len(dicti[dict_key])):
        if dicti['pred'][i] == pred_type:
            output.append(dicti[dict_key][i])
    return output

def read_fits(filename):

    hdu = fits.open("4MOST_CatForGregoire_18Sep2018.fits")
    data = hdu[1].data
    data_keys = hdu[1].columns.names

    return data, data_keys

def reduce_dict(data_object, wanted_keys):

    for i in list(data_object.keys()):
        if not any(substring in i for substring in wanted_keys):
            del data_object[i]
            continue
        if 'id' in i:
            if 'DES_id' in i:
                continue
            else:
                del data_object[i]
                continue
        if i in ['isWISE', 'isVHS', 'DES_ra', 'DES_dec']:
            del data_object[i]
            continue

    return data_object

def ELG_selector(data, keys):

    x = data['DES_g']-data['DES_r']
    y = data['DES_r']-data['DES_i']
    iselg = (
            (data['HSC_zphot']>0.7) &
            (data['HSC_zphot']<1.1) &
            (21<data['DES_g']) &
            (data['DES_g']<23.2) &
            (0.5-2.5*x<y) &
            (y<3.5-2.5*x) &
            (0.4*x+0.3<y) &
            (y<0.4*x+0.9)
            )
    ELG_dict = {n:data[n][iselg] for n in keys}
    return ELG_dict, len(data[iselg])

def LRG_selector(data, keys):

    x = data['VHS_j']-data['VHS_k']
    y = data['VHS_j']-data['WISE_w1']
    islrg = (
            (data['isVHS']==True) &
            (data['isWISE']==True) &
            (data['HSC_zphot']>0.4) &
            (data['HSC_zphot']<0.8) &
            (18<data['VHS_j']) &
            (data['VHS_j']<19.5) &
            (data['VHS_class']==1) &
            (x>0.25) &
            (x<1.50) &
            (y<1.50) &
            (y>1.6*x-1.5) &
            (y>-0.5*x+0.65)
            )
    LRG_dict = {n:data[n][islrg] for n in keys}
    return LRG_dict, len(data[islrg])

def BG_selector(data, keys):

    x = data['VHS_j']-data['VHS_k']
    y = data['VHS_j']-data['WISE_w1']
    isbg = (
           (data['isVHS']==True) &
           (data['isWISE']==True) &
           (data['HSC_zphot']>0.05) &
           (data['HSC_zphot']<0.4) &
           (16<data['VHS_j']) &
           (data['VHS_j']<18) &
           (data['VHS_class']==1) &
           (x>0.10) &
           (x<1.00) &
           (y>1.6*x-1.6) &
           (y<1.6*x-0.5) &
           (y>-0.5*x-1.0) &
           (y<-0.5*x+0.1)
           )
    BG_dict = {n:data[n][isbg] for n in keys}
    return BG_dict, len(data[isbg])

def QSO_selector(data, keys):

    isqso = (
            (data['isWISE']==True) &
            (data['isDR14QSO']==True)
            )
    QSO_dict = {n:data[n][isqso] for n in keys}
    return QSO_dict, len(data[isqso])

def Other_selector(data, keys):

    x = data['DES_g']-data['DES_r']
    y = data['DES_r']-data['DES_i']
    iselg = (
            (data['HSC_zphot']>0.7) &
            (data['HSC_zphot']<1.1) &
            (21<data['DES_g']) &
            (data['DES_g']<23.2) &
            (0.5-2.5*x<y) &
            (y<3.5-2.5*x) &
            (0.4*x+0.3<y) &
            (y<0.4*x+0.9)
            )
    notelg = [not i for i in iselg]
    x = data['VHS_j']-data['VHS_k']
    y = data['VHS_j']-data['WISE_w1']
    islrg = (
            (data['isVHS']==True) &
            (data['isWISE']==True) &
            (data['HSC_zphot']>0.4) &
            (data['HSC_zphot']<0.8) &
            (18<data['VHS_j']) &
            (data['VHS_j']<19.5) &
            (data['VHS_class']==1) &
            (x>0.25) &
            (x<1.50) &
            (y<1.50) &
            (y>1.6*x-1.5) &
            (y>-0.5*x+0.65)
            )
    notlrg = [not i for i in islrg]
    x = data['VHS_j']-data['VHS_k']
    y = data['VHS_j']-data['WISE_w1']
    isbg = (
           (data['isVHS']==True) &
           (data['isWISE']==True) &
           (data['HSC_zphot']>0.05) &
           (data['HSC_zphot']<0.4) &
           (16<data['VHS_j']) &
           (data['VHS_j']<18) &
           (data['VHS_class']==1) &
           (x>0.10) &
           (x<1.00) &
           (y>1.6*x-1.6) &
           (y<1.6*x-0.5) &
           (y>-0.5*x-1.0) &
           (y<-0.5*x+0.1)
           )
    notbg = [not i for i in isbg]
    isqso = (
            (data['isWISE']==True) &
            (data['isDR14QSO']==True)
            )
    notqso = [not i for i in isqso]
    isothers = []
    for i in range(len(isqso)):
        isothers.append(not any([iselg[i], isbg[i], islrg[i], isqso[i]]))
    Others_dict = {n:data[n][isothers] for n in keys}
    return Others_dict, len(data[isothers])

def fill_empty_entries(object_dict_list, arbitrary_magnitude):

    all_WISE_w1err = []
    all_VHS_jerr = []
    all_VHS_kerr = []

    for i in object_dict_list:
        for j in ['VHS_j','VHS_k', 'VHS_jerr', 'VHS_kerr', 'WISE_w1', 'WISE_w1err']:
            for k in i[j]:
                if not np.isnan(k):
                    if 'jerr' in j:
                        all_VHS_jerr.append(k)
                    elif 'w1err' in j:
                        all_WISE_w1err.append(k)
                    elif 'kerr' in j:
                        all_VHS_kerr.append(k)

    mean_w1err = sum(all_WISE_w1err)/float(len(all_WISE_w1err))
    mean_jerr = sum(all_VHS_jerr)/float(len(all_VHS_jerr))
    mean_kerr = sum(all_VHS_kerr)/float(len(all_VHS_kerr))

    for i in object_dict_list:
        for j in ['VHS_j','VHS_k', 'VHS_jerr', 'VHS_kerr', 'WISE_w1', 'WISE_w1err']:
            for idx, k in enumerate(i[j]):
                if (np.isnan(k)) and ('err' in j):
                    if 'jerr' in j:
                        i[j][idx] = mean_jerr
                    elif 'w1err' in j:
                        i[j][idx] = mean_w1err
                    elif 'kerr' in j:
                        i[j][idx] = mean_kerr
                elif np.isnan(k):
                    i[j][idx] = arbitrary_magnitude

    return object_dict_list

def full_multiclass_dataset(full_data):

    data, data_keys = read_fits("4MOST_CatForGregoire_18Sep2018.fits")
    wanted_keys = ['DES', 'VHS', 'WISE', 'HSC_zphot', 'VHS_class']

    elgs, nbrelg = ELG_selector(data, data_keys)
    lrgs, nbrlrg = LRG_selector(data, data_keys)
    bgs, nbrbg = BG_selector(data, data_keys)
    qsos, nbrqso = QSO_selector(data, data_keys)
    others, nbrothers = Other_selector(data, data_keys)

    nbr_objects = nbrelg + nbrlrg + nbrbg + nbrqso + nbrothers

    reduced_elgs = reduce_dict(elgs, wanted_keys)
    reduced_lrgs = reduce_dict(lrgs, wanted_keys)
    reduced_bgs = reduce_dict(bgs, wanted_keys)
    reduced_qsos = reduce_dict(qsos, wanted_keys)
    reduced_others = reduce_dict(others, wanted_keys)

    reduced_others['class'] = [0 for i in range(nbrothers)]
    reduced_elgs['class'] = [1 for i in range(nbrelg)]
    reduced_lrgs['class'] = [2 for i in range(nbrlrg)]
    reduced_bgs['class'] = [3 for i in range(nbrbg)]
    reduced_qsos['class'] = [4 for i in range(nbrqso)]

    if full_data:
        all_objects = [reduced_others, reduced_elgs, reduced_lrgs, reduced_bgs, reduced_qsos]
    else:
        all_objects = [reduced_elgs, reduced_lrgs, reduced_bgs, reduced_qsos]

    magnitude_correction = 25.0
    all_objects_filled =  fill_empty_entries(all_objects, magnitude_correction)

    reduced_keys = list(reduced_elgs.keys())
    nbr_features = len(reduced_keys)

    print(reduced_keys)

    full_dataset = []

    for i in range(nbr_features):
        tmp_list = []
        for j in all_objects:
            tmp_list += list(j[reduced_keys[i]])
        full_dataset.append(tmp_list)

    full_table_dataset = np.transpose(np.array(full_dataset, dtype='float'))

    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    if full_data:
        savepath = os.path.join(script_dir, 'datasets', 'full', 'histograms.csv')
        np.savetxt(savepath, full_table_dataset, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding=None)
    else:
        savepath = os.path.join(script_dir, 'datasets', 'notfull', 'histograms.csv')
        np.savetxt("DES_elg_binary_dataset.csv", full_table_dataset, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding=None)
    predictor(full_data, reduced_keys)

def predictor(full_data, reduced_keys):
    dataset_name = 'histograms.csv'
    model_name = '4layers_softmax-ouputactivation_2000batch50epochs'
    folder_name = 'BG_ELG_LRG_QSO_classification'
    classnames = ['Other', 'ELG', 'LRG', 'BG', 'QSO']

    model_arch_name = model_name + '.json'
    model_weights_name = model_name + '_final.h5'
    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    if full_data:
        dataset_path = os.path.join(script_dir, 'datasets', 'full', dataset_name)
        model_architecture_path = os.path.join(script_dir, 'model', folder_name, 'full', model_arch_name)
        model_weights_path = os.path.join(script_dir, 'model', folder_name, 'full', model_weights_name)
    else:
        dataset_path = os.path.join(script_dir, 'datasets','notfull', dataset_name)
        model_architecture_path = os.path.join(script_dir, 'model', folder_name, 'notfull', model_arch_name)
        model_weights_path = os.path.join(script_dir, 'model', folder_name, 'notfull', model_weights_name)

    # load dataset
    dataset = np.loadtxt(dataset_path, delimiter=",")
    np.random.shuffle(dataset)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # split into input (X) and output (Y) variables
    X = dataset[:,:-1]
    Y = dataset[:,-1]
    input_dimensions = X.shape[1]

    # One hot encoding of labels contained in Y
    Y = np.array(Y, dtype='int')
    n_classes = len(set(Y))
    Y = Y.reshape(len(Y), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_onehot_encoded = onehot_encoder.fit_transform(Y)

    # Split in testing and training dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y_onehot_encoded, test_size=0.20, random_state=seed)


    X_test_part = X_test[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20]]
    # load json and create model
    json_file = open(model_architecture_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    y_score = loaded_model.predict(X_test_part)

    ELG_dict = {'pred': []}
    BG_dict = {'pred': []}
    QSO_dict = {'pred': []}
    LRG_dict = {'pred': []}
    Other_dict = {'pred': []}
    all_dict = [Other_dict, ELG_dict, LRG_dict, BG_dict, QSO_dict]
    y_pred_labels = y_score.argmax(axis=-1)
    y_true_labels = y_test.argmax(axis=-1)
    print(list(set(y_pred_labels)))
    print(list(set(y_true_labels)))

    for i in range(len(y_true_labels)):
        if y_pred_labels[i] == y_true_labels[i]:
            all_dict[y_true_labels[i]]['pred'].append('TP')
            for j in range(input_dimensions):
                if not reduced_keys[j] in list(all_dict[y_true_labels[i]].keys()):
                    all_dict[y_true_labels[i]][reduced_keys[j]] = [X_test[i, j]]
                else:
                    all_dict[y_true_labels[i]][reduced_keys[j]].append(X_test[i, j])
        else:
            all_dict[y_pred_labels[i]]['pred'].append('FP')
            all_dict[y_true_labels[i]]['pred'].append('FN')
            for j in range(input_dimensions):
                if not reduced_keys[j] in list(all_dict[y_true_labels[i]].keys()):
                    all_dict[y_true_labels[i]][reduced_keys[j]] =  [X_test[i, j]]
                else:
                    all_dict[y_true_labels[i]][reduced_keys[j]].append(X_test[i, j])
                if not reduced_keys[j] in list(all_dict[y_pred_labels[i]].keys()):
                    all_dict[y_pred_labels[i]][reduced_keys[j]] =  [X_test[i, j]]
                else:
                    all_dict[y_pred_labels[i]][reduced_keys[j]].append(X_test[i, j])

    all_bg_rmag_tp = extract_keys_from_pred(all_dict[3], 'DES_r', 'TP')
    all_bg_rmag_fp = extract_keys_from_pred(all_dict[3], 'DES_r', 'FP')
    all_bg_rmag_fn = extract_keys_from_pred(all_dict[3], 'DES_r', 'FN')

    all_lrg_rmag_tp = extract_keys_from_pred(all_dict[2], 'DES_r', 'TP')
    all_lrg_rmag_fp = extract_keys_from_pred(all_dict[2], 'DES_r', 'FP')
    all_lrg_rmag_fn = extract_keys_from_pred(all_dict[2], 'DES_r', 'FN')

    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax0, ax1 = axes.flatten()

    colors = ['blue', 'red', 'green']
    labels = ['True Positive','False Positive', 'False Negative']

    n_bins_rmag = 10
    ax0.hist([all_bg_rmag_tp, all_bg_rmag_fp, all_bg_rmag_fn], n_bins_rmag, density=True, histtype='bar', color=colors, label=labels)
    ax0.legend(prop={'size': 10})
    ax0.set_title('DES_r mag for BG')

    n_bins_rmag = 10
    ax1.hist([all_lrg_rmag_tp, all_lrg_rmag_fp, all_lrg_rmag_fn], n_bins_rmag, density=True, histtype='bar', color=colors, label=labels)
    ax1.legend(prop={'size': 10})
    ax1.set_title('DES_r mag for LRG')

    # n_bins_VHS_CLASS = 4
    # ax0.hist(x, n_bins_VHS_CLASS, density=True, histtype='bar', color=colors, label=colors)
    # ax0.legend(prop={'size': 10})
    # ax0.set_title('bars with legend')

    fig.tight_layout()
    plt.show()
full_data = True
full_multiclass_dataset(full_data)
