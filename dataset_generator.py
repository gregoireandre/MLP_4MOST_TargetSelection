import numpy as np
import math
import os
import itertools
import operator
from astropy.io import fits

#repr(fits_header)  # Representation of the fits header
#list(fits_header.keys()) # Prints a representation of the fits header
#fits_header[0] # Get first keys
#fits_data[fits_header_key] #Returns fits data corresponding to the fits_header_key passed as dictionnary key

def generate_all_datasets():

    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    fits_filename = "4MOST.CatForGregoire.11Oct2018.zphot.fits"
    cv_ratios = [[90, 10, 10], [80, 20, 5]]
    constraints = 'no'
    all_constraints_str = constraints_to_str(constraints)

    others_flag = 'all'
    constraints = 'no'
    all_constraints_str = constraints_to_str(constraints)
    full_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)
    noqso_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)
    object_to_select = 'BG'
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'ELG'
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'LRG'
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'QSO'
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'BG'
    constraints = {'16J18': [['16', '<', 'VHS_j'], #BG Cut
                             ['18', '>', 'VHS_j']]
                  }
    all_constraints_str = constraints_to_str(constraints)
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'LRG'
    constraints = {'18J19.5': [['18', '<', 'VHS_j'], #LRG Cut
                             ['19.5', '>', 'VHS_j']]
                  }
    all_constraints_str = constraints_to_str(constraints)
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)

    others_flag = 'no'
    constraints = 'no'
    all_constraints_str = constraints_to_str(constraints)
    full_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)
    noqso_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)
    object_to_select = 'BG'
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'ELG'
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'LRG'
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'QSO'
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'BG'
    constraints = {'16J18': [['16', '<', 'VHS_j'], #BG Cut
                             ['18', '>', 'VHS_j']]
                  }
    all_constraints_str = constraints_to_str(constraints)
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    object_to_select = 'LRG'
    constraints = {'18J19.5': [['18', '<', 'VHS_j'], #LRG Cut
                             ['19.5', '>', 'VHS_j']]
                  }
    all_constraints_str = constraints_to_str(constraints)
    binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)

    # fits_filename = "4MOST.CatForGregoire.02Oct2018.fits"
    # others_flag = 'all'
    # constraints = 'no'
    # all_constraints_str = constraints_to_str(constraints)
    # full_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)
    # noqso_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)
    # object_to_select = 'BG'
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'ELG'
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'LRG'
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'QSO'
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'BG'
    # constraints = {'16J18': [['16', '<', 'VHS_j'], #BG Cut
    #                          ['18', '>', 'VHS_j']]
    #               }
    # all_constraints_str = constraints_to_str(constraints)
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'LRG'
    # constraints = {'18J19.5': [['18', '<', 'VHS_j'], #LRG Cut
    #                          ['19.5', '>', 'VHS_j']]
    #               }
    # all_constraints_str = constraints_to_str(constraints)
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    #
    # others_flag = 'no'
    # constraints = 'no'
    # all_constraints_str = constraints_to_str(constraints)
    # full_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)
    # noqso_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)
    # object_to_select = 'BG'
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'ELG'
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'LRG'
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'QSO'
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'BG'
    # constraints = {'16J18': [['16', '<', 'VHS_j'], #BG Cut
    #                          ['18', '>', 'VHS_j']]
    #               }
    # all_constraints_str = constraints_to_str(constraints)
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)
    # object_to_select = 'LRG'
    # constraints = {'18J19.5': [['18', '<', 'VHS_j'], #LRG Cut
    #                          ['19.5', '>', 'VHS_j']]
    #               }
    # all_constraints_str = constraints_to_str(constraints)
    # binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)

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

def apply_constraints(constraints, data):

    ops = { "<": operator.lt, ">": operator.gt, "==": operator.eq}

    for i in list(constraints.keys()):
        for j in constraints[i]:
            if 'isinconstraint' in locals():
                isinconstraint &= ops[j[1]](float(j[0]), data[j[2]])
            else:
                isinconstraint = ops[j[1]](float(j[0]), data[j[2]])
    constrained_data = data[isinconstraint]
    return constrained_data

def read_fits(path, filename):

    ####role####
    # Read a .fits file and returns the data and keys contained in the latter.

    ####inputs####
    # path : path of the .fits file to load
    # filename : filename of the .fits file to load

    ####outputs####
    # data_keys : The keys corresponding to the data inside the .fits file (eg DES_r, HSC_zphot, ...)
    # data : Object containing the data of the .fits file. The object has the same behavior than
    # python dictionnary (eg to access all the DES_r data just type data['DES_r'])

    hdu = fits.open(os.path.join(path, filename))
    data = hdu[1].data
    data_keys = hdu[1].columns.names

    return data, data_keys

def ELG_selector(data, keys):

    ####role####
    # Select ELGs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

    ####inputs####
    # data : Should be the data object outputed by the read_fits function which contains all the data
    # from the .fits file.
    # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

    ####outputs####
    # ELG_dict : A python dictionnary containing all the ELG's data ordered within the same keys as in the data object taken as input.
    # nbr_elg : The number of ELG that were selected

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
    nbr_elg = len(data[iselg])
    return ELG_dict, nbr_elg

def LRG_selector(data, keys):

    ####role####
    # Select LRGs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

    ####inputs####
    # data : Should be the data object outputed by the read_fits function which contains all the data
    # from the .fits file.
    # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

    ####outputs####
    # LRG_dict : A python dictionnary containing all the LRG's data ordered within the same keys as in the data object taken as input.
    # nbr_lrg : The number of LRG that were selected

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
    nbr_lrg = len(data[islrg])
    return LRG_dict, nbr_lrg

def BG_selector(data, keys):

    ####role####
    # Select BGs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

    ####inputs####
    # data : Should be the data object outputed by the read_fits function which contains all the data
    # from the .fits file.
    # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

    ####outputs####
    # BG_dict : A python dictionnary containing all the bg's data ordered within the same keys as in the data object taken as input.
    # nbr_bg : The number of BG that were selected

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
    nbr_bg = len(data[isbg])
    return BG_dict, nbr_bg

def QSO_selector(data, keys):

    ####role####
    # Select QSOs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

    ####inputs####
    # data : Should be the data object outputed by the read_fits function which contains all the data
    # from the .fits file.
    # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

    ####outputs####
    # QSO_dict : A python dictionnary containing all the QSO's data ordered within the same keys as in the data object taken as input.
    # nbr_qso : The number of QSO that were selected

    isqso = (
            (data['isWISE']==True) &
            (data['isDR14QSO']==True)
            )
    QSO_dict = {n:data[n][isqso] for n in keys}
    nbr_qso = len(data[isqso])
    return QSO_dict, nbr_qso

def Other_selector(data, keys):

    ####role####
    # Select all entry that do not correspond to ELG, LRG, BG, QSO (eg "other") from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

    ####inputs####
    # data : Should be the data object outputed by the read_fits function which contains all the data
    # from the .fits file.
    # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

    ####outputs####
    # Others_dict : A python dictionnary containing all the other's data ordered within the same keys as in the data object taken as input.
    # nbr_other : The number of "other" that were selected

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
    nbr_other = len(data[isothers])
    return Others_dict, nbr_other

def reduce_dict(data_dict, wanted_keys, exceptions_keys):

    ####role####
    # Reduce the dimension of a data object by keeping only keys of interest.

    ####inputs####
    # data_dict : Should be a python dictionnnary containing data of a particular type of object such as ELG, LRG, BG, QSO or Other
    # wanted_keys : List of keys (eg string) that we want to keep in the data object. One should note that the keys of interest are
    # selected using the 'in' python operator. This means that all the keys containing one of the element of wanted_keys will be kept
    # in the data object. As an exeample, if 'DES' is in wanted_keys, all the DES data (eg 'DES_r', DES_g', ...) will be kept in the data object.
    # exceptions_keys : List of keys to remove from the dictionnary, it allows to raise exceptions on the keys selection. As stated above, if one wants to keep all
    # the DES data apart from the 'DES_ra' and 'DES_dec', it is more conveninet to keep all 'DES' data and raise an exception for those two keys than to specify
    # all the other 'DES' keys that are wanted.

    ####outputs####
    # data_dict : The reduced data_dict where all the keys that were not included in wanted_keys and those that were included in
    # exceptions_keys were removed

    for i in list(data_dict.keys()):
        if not any(substring in i for substring in wanted_keys):
            del data_dict[i]
            continue
        if 'DES_id' in i:
            continue
        # ID's are not relevant informations so we delete it
        if 'id' in i:
            del data_dict[i]
            continue
        if i in exceptions_keys:
            del data_dict[i]
            continue

    return data_dict

def fill_empty_entries(object_dict_list, arbitrary_magnitude):

    ####role####
    # Fill empty entries in the dataset. In our case, there are empty entries for the

    ####inputs####
    # data : Should be the data object outputed by the read_fits function which contains all the data
    # from the .fits file.
    # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

    ####outputs####
    # QSO_dict : A python dictionnary containing all the QSO's data ordered within the same keys as in the data object taken as input.
    # nbr_qso : The number of QSO that were selected

    # all_WISE_w1err = []
    # all_VHS_jerr = []
    # all_VHS_kerr = []
    #
    # for i in object_dict_list:
    #     for j in ['VHS_j','VHS_k', 'VHS_jerr', 'VHS_kerr', 'WISE_w1', 'WISE_w1err']:
    #         for k in i[j]:
    #             if not np.isnan(k):
    #                 if 'jerr' in j:
    #                     all_VHS_jerr.append(k)
    #                 elif 'w1err' in j:
    #                     all_WISE_w1err.append(k)
    #                 elif 'kerr' in j:
    #                     all_VHS_kerr.append(k)
    #
    # mean_w1err = sum(all_WISE_w1err)/float(len(all_WISE_w1err))
    # mean_jerr = sum(all_VHS_jerr)/float(len(all_VHS_jerr))
    # mean_kerr = sum(all_VHS_kerr)/float(len(all_VHS_kerr))
    #
    # for i in object_dict_list:
    #     for j in ['VHS_j','VHS_k', 'VHS_jerr', 'VHS_kerr', 'WISE_w1', 'WISE_w1err']:
    #         for idx, k in enumerate(i[j]):
    #             if (np.isnan(k)) and ('err' in j):
    #                 if 'jerr' in j:
    #                     i[j][idx] = mean_jerr
    #                 elif 'w1err' in j:
    #                     i[j][idx] = mean_w1err
    #                 elif 'kerr' in j:
    #                     i[j][idx] = mean_kerr
    #             elif np.isnan(k):
    #                 i[j][idx] = arbitrary_magnitude

    for i in object_dict_list:
        for j in ['VHS_j','VHS_k', 'VHS_jerr', 'VHS_kerr', 'WISE_w1', 'WISE_w1err']:
            for idx, k in enumerate(i[j]):
                if np.isnan(k):
                    i[j][idx] = arbitrary_magnitude

    return object_dict_list

def generate_cross_validation_datasets(cv_ratios, class_dict, dataset_np_array, reduced_keys, filename, others_flag, constraints, fits_filename):

    nbr_objects = dataset_np_array.shape[0]
    nbr_features = dataset_np_array.shape[1]
    nbr_classes = len(class_dict['labels'])
    isclass = {}
    for label in class_dict['labels']:
        isclass[label] = (dataset_np_array[:,-1]==label)
    print(isclass)

    for i in cv_ratios:
        train_ratio = i[0]/100
        test_ratio = i[1]/100
        nbr_fold = i[2]
        indexes_chosen_test = np.zeros(nbr_objects,dtype=bool)
        for j in range(nbr_fold):
            mask_train = np.ones(nbr_objects,dtype=bool)
            table_dataset_train = []
            idx_test = []
            nbr_object_test = []
            for h in class_dict['nbr_objects']:
                nbr_object_test.append(math.floor(h*test_ratio))
            print('Use ', i[1], ' percent of data : ', nbr_object_test)
            print('Full data is ', class_dict['nbr_objects'])
            for labidx, label in enumerate(class_dict['labels']):
                avail = np.where((isclass[label]) &  (~indexes_chosen_test))[0]
                idx_test += avail[:nbr_object_test[labidx]].tolist()
                indexes_chosen_test[avail[:nbr_object_test[labidx]]] = True
                mask_train[avail[:nbr_object_test[labidx]]] = False
            idx_train = np.where(mask_train)[0]
            table_dataset_test = dataset_np_array[idx_test, :]
            table_dataset_train= dataset_np_array[idx_train,:]
            #     if (nbr_object_test[l] >= 1) and (dataset_np_array[k][-1] == class_dict['labels'][l]) and (k not in indexes_chosen_test):
            #         idx_test.append(k)
            #         nbr_object_test[l] -= 1
            # indexes_chosen_test += idx_test
            # idx_train = [m for m in range(nbr_objects) if m not in idx_test]
            # table_dataset_train = dataset_np_array[idx_train, :]
            # table_dataset_test = dataset_np_array[idx_test, :]
            savepath = os.path.join(script_dir, 'datasets', fits_filename, others_flag + '-others_' + constraints + '-constraints')
            all_fits_column_test = []
            all_fits_column_train = []
            print(table_dataset_test)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            for g in range(nbr_features):
                all_fits_column_train.append(fits.Column(name=reduced_keys[g], array=table_dataset_train[:,g], format='D'))
                all_fits_column_test.append(fits.Column(name=reduced_keys[g], array=table_dataset_test[:,g], format='D'))
            test = fits.BinTableHDU.from_columns(all_fits_column_test)
            train = fits.BinTableHDU.from_columns(all_fits_column_train)
            try:
                test.writeto(os.path.join(savepath, filename + '_test_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits'))
                print(filename + '_test_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits')
            except OSError as e:
                print(e)
            try:
                train.writeto(os.path.join(savepath, filename + '_train_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits'))
                print(filename + '_train_' + str(i[0]) + '_'+ str(i[1]) + '_' + str(j+1) + '.fits')
            except OSError as e:
                print(e)

    return

def check_dataset_validity(cv_ratios, fits_filename, constraints, others_flag, classification_problem):

    path = os.path.join(script_dir, 'datasets', fits_filename, others_flag + '-others_' + all_constraints_str + '-constraints')

    dataset_is_valid = False
    all_cv_test_datasets = []
    all_test = {}
    for i in cv_ratios:
        for j in range(i[2]):
            tmp, _ = read_fits(path, classification_problem + '_test_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits')
            all_cv_test_datasets.append(tmp)
        all_sets = [i+1 for i in np.arange(i[2])]
        all_combinations = itertools.combinations(all_sets, 2)
        for k in all_combinations:
            row_comparison = []
            for l in range(all_cv_test_datasets[0].shape[0]):
                row_comparison.append(all(np.equal(all_cv_test_datasets[k[0] - 1][l],all_cv_test_datasets[k[1] - 1][l])))
            if i[0] not in list(all_test.keys()):
                all_test[i[0]] = [ [k[0], k[1], any(row_comparison)] ]
            else:
                all_test[i[0]].append([k[0], k[1], any(row_comparison)])
    for i in list(all_test.keys()):
        for j in all_test[i]:
            print('Test on ', fits_filename + '_test_' + str(i) + '_' + str(j[0]) + ' and ' +  fits_filename + '_test_' + str(i) + '_' + str(j[1]), ' gives : ', j[2])
    return

def labelize(reduced_elgs, nbrelg, reduced_lrgs, nbrlrg, reduced_bgs, nbrbg, reduced_qsos, nbrqso, reduced_others, nbrothers, object_to_select):

    if object_to_select == 'ELG':

        reduced_elgs['class'] = [1 for i in range(nbrelg)]
        reduced_lrgs['class'] = [0 for i in range(nbrlrg)]
        reduced_bgs['class'] = [0 for i in range(nbrbg)]
        reduced_qsos['class'] = [0 for i in range(nbrqso)]
        reduced_others['class'] = [0 for i in range(nbrothers)]

    elif object_to_select == 'LRG':

        reduced_elgs['class'] = [0 for i in range(nbrelg)]
        reduced_lrgs['class'] = [1 for i in range(nbrlrg)]
        reduced_bgs['class'] = [0 for i in range(nbrbg)]
        reduced_qsos['class'] = [0 for i in range(nbrqso)]
        reduced_others['class'] = [0 for i in range(nbrothers)]

    elif object_to_select == 'BG':

        reduced_elgs['class'] = [0 for i in range(nbrelg)]
        reduced_lrgs['class'] = [0 for i in range(nbrlrg)]
        reduced_bgs['class'] = [1 for i in range(nbrbg)]
        reduced_qsos['class'] = [0 for i in range(nbrqso)]
        reduced_others['class'] = [0 for i in range(nbrothers)]

    elif object_to_select == 'QSO':

        reduced_elgs['class'] = [0 for i in range(nbrelg)]
        reduced_lrgs['class'] = [0 for i in range(nbrlrg)]
        reduced_bgs['class'] = [0 for i in range(nbrbg)]
        reduced_qsos['class'] = [1 for i in range(nbrqso)]
        reduced_others['class'] = [0 for i in range(nbrothers)]

    elif object_to_select == 'Others':

        reduced_elgs['class'] = [1 for i in range(nbrelg)]
        reduced_lrgs['class'] = [1 for i in range(nbrlrg)]
        reduced_bgs['class'] = [1 for i in range(nbrbg)]
        reduced_qsos['class'] = [1 for i in range(nbrqso)]
        reduced_others['class'] = [0 for i in range(nbrothers)]

    return reduced_elgs, reduced_lrgs, reduced_bgs, reduced_qsos, reduced_others

def noqso_multiclass_dataset(fits_path, fits_filename, others_flag, constraints):

    data, data_keys = read_fits(fits_path, fits_filename)
    fits_filename, _ = os.path.splitext(fits_filename)

    if constraints != 'no':
        data = apply_constraints(constraints, data)
        all_constraints_str = ''
        for s in list(constraints.keys()):
            if all_constraints_str:
                all_constraints_str += '+' + s
            else:
                all_constraints_str = s
    else:
        all_constraints_str = 'no'

    elgs, nbrelg = ELG_selector(data, data_keys)
    lrgs, nbrlrg = LRG_selector(data, data_keys)
    bgs, nbrbg = BG_selector(data, data_keys)
    qsos, nbrqso = QSO_selector(data, data_keys)
    others, nbrothers = Other_selector(data, data_keys)

    nbr_objects = nbrelg + nbrlrg + nbrbg + nbrqso + nbrothers

    wanted_keys = ['DES', 'VHS', 'WISE']
    exceptions_keys = ['isWISE', 'isVHS', 'VHS_class','DES_ra', 'DES_dec', 'DES_spread']

    reduced_elgs = reduce_dict(elgs, wanted_keys, exceptions_keys)
    reduced_lrgs = reduce_dict(lrgs, wanted_keys, exceptions_keys)
    reduced_bgs = reduce_dict(bgs, wanted_keys, exceptions_keys)
    reduced_qsos = reduce_dict(qsos, wanted_keys, exceptions_keys)
    reduced_others = reduce_dict(others, wanted_keys, exceptions_keys)

    reduced_qsos['class'] = [0 for i in range(nbrqso)]
    reduced_others['class'] = [0 for i in range(nbrothers)]
    reduced_elgs['class'] = [1 for i in range(nbrelg)]
    reduced_lrgs['class'] = [2 for i in range(nbrlrg)]
    reduced_bgs['class'] = [3 for i in range(nbrbg)]

    if others_flag == 'all':
        all_objects = [reduced_qsos, reduced_others, reduced_elgs, reduced_lrgs, reduced_bgs]
    else:
        all_objects = [reduced_qsos, reduced_elgs, reduced_lrgs, reduced_bgs]

    magnitude_correction = 0.0
    all_objects_filled =  fill_empty_entries(all_objects, magnitude_correction)

    reduced_keys = list(reduced_elgs.keys())
    nbr_features = len(reduced_keys)

    full_dataset = []

    for i in range(nbr_features):
        tmp_list = []
        for j in all_objects:
            tmp_list += list(j[reduced_keys[i]])
        full_dataset.append(tmp_list)

    full_table_dataset = np.transpose(np.array(full_dataset, dtype='float'))

    savepath = os.path.join(script_dir, 'datasets', fits_filename, others_flag + '-others_' + all_constraints_str + '-constraints')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataset_filename = 'BG_ELG_LRG_classification'

    all_fits_column = []
    for i in range(nbr_features):
        all_fits_column.append(fits.Column(name=reduced_keys[i], array=full_table_dataset[:,i], format='D'))
    dataset = fits.BinTableHDU.from_columns(all_fits_column)

    try:
        dataset.writeto(os.path.join(savepath, dataset_filename + '.fits'))
    except OSError as e:
        print(e)

    if others_flag == 'all':
        class_dict = {'nbr_objects': [nbrqso + nbrothers, nbrelg, nbrlrg, nbrbg], 'labels': [0, 1, 2, 3]}
    else:
        class_dict = {'nbr_objects': [nbrqso, nbrelg, nbrlrg, nbrbg], 'labels': [0, 1, 2, 3]}

    generate_cross_validation_datasets(cv_ratios, class_dict, full_table_dataset, reduced_keys, dataset_filename, others_flag, all_constraints_str, fits_filename)

def full_multiclass_dataset(fits_path, fits_filename, others_flag, constraints):

    data, data_keys = read_fits(fits_path, fits_filename)
    fits_filename, _ = os.path.splitext(fits_filename)

    elgs, nbrelg = ELG_selector(data, data_keys)
    lrgs, nbrlrg = LRG_selector(data, data_keys)
    bgs, nbrbg = BG_selector(data, data_keys)
    qsos, nbrqso = QSO_selector(data, data_keys)
    others, nbrothers = Other_selector(data, data_keys)

    nbr_objects = nbrelg + nbrlrg + nbrbg + nbrqso + nbrothers

    wanted_keys = ['DES', 'VHS', 'WISE']
    exceptions_keys = ['isWISE', 'isVHS', 'VHS_class','DES_ra', 'DES_dec', 'DES_spread']

    reduced_others = reduce_dict(others, wanted_keys, exceptions_keys)
    reduced_elgs = reduce_dict(elgs, wanted_keys, exceptions_keys)
    reduced_lrgs = reduce_dict(lrgs, wanted_keys, exceptions_keys)
    reduced_bgs = reduce_dict(bgs, wanted_keys, exceptions_keys)
    reduced_qsos = reduce_dict(qsos, wanted_keys, exceptions_keys)

    reduced_others['class'] = [0 for i in range(nbrothers)]
    reduced_elgs['class'] = [1 for i in range(nbrelg)]
    reduced_lrgs['class'] = [2 for i in range(nbrlrg)]
    reduced_bgs['class'] = [3 for i in range(nbrbg)]
    reduced_qsos['class'] = [4 for i in range(nbrqso)]

    if others_flag == 'all':
        all_objects = [reduced_others, reduced_elgs, reduced_lrgs, reduced_bgs, reduced_qsos]
    else:
        all_objects = [reduced_elgs, reduced_lrgs, reduced_bgs, reduced_qsos]

    magnitude_correction = 0.0
    all_objects_filled =  fill_empty_entries(all_objects, magnitude_correction)

    reduced_keys = list(reduced_elgs.keys())
    nbr_features = len(reduced_keys)

    full_dataset = []

    for i in range(nbr_features):
        tmp_list = []
        for j in all_objects:
            tmp_list += list(j[reduced_keys[i]])
        full_dataset.append(tmp_list)

    full_table_dataset = np.transpose(np.array(full_dataset, dtype='float'))

    savepath = os.path.join(script_dir, 'datasets', fits_filename, others_flag + '-others_' + all_constraints_str + '-constraints')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataset_filename = 'BG_ELG_LRG_QSO_classification'

    all_fits_column = []
    for i in range(nbr_features):
        all_fits_column.append(fits.Column(name=reduced_keys[i], array=full_table_dataset[:,i], format='D'))
    dataset = fits.BinTableHDU.from_columns(all_fits_column)

    try:
        dataset.writeto(os.path.join(savepath, dataset_filename + '.fits'))
    except OSError as e:
        print(e)

    if others_flag == 'all':
        class_dict = {'nbr_objects': [nbrothers, nbrelg, nbrlrg, nbrbg, nbrqso], 'labels': [0, 1, 2, 3, 4]}
    else:
        class_dict = {'nbr_objects': [nbrelg, nbrlrg, nbrbg, nbrqso], 'labels': [1, 2, 3, 4]}

    generate_cross_validation_datasets(cv_ratios, class_dict, full_table_dataset, reduced_keys, dataset_filename, others_flag, all_constraints_str, fits_filename)

def noelg_multiclass_dataset(fits_path, fits_filename, others_flag, constraints):

    data, data_keys = read_fits(fits_path, fits_filename)
    fits_filename, _ = os.path.splitext(fits_filename)

    lrgs, nbrlrg = LRG_selector(data, data_keys)
    bgs, nbrbg = BG_selector(data, data_keys)
    qsos, nbrqso = QSO_selector(data, data_keys)
    others, nbrothers = Other_selector(data, data_keys)

    nbr_objects = nbrlrg + nbrbg + nbrqso + nbrothers

    wanted_keys = ['DES', 'VHS', 'WISE']
    exceptions_keys = ['isWISE', 'isVHS', 'VHS_class','DES_ra', 'DES_dec', 'DES_spread']

    reduced_others = reduce_dict(others, wanted_keys, exceptions_keys)
    reduced_lrgs = reduce_dict(lrgs, wanted_keys, exceptions_keys)
    reduced_bgs = reduce_dict(bgs, wanted_keys, exceptions_keys)
    reduced_qsos = reduce_dict(qsos, wanted_keys, exceptions_keys)

    reduced_others['class'] = [0 for i in range(nbrothers)]
    reduced_lrgs['class'] = [1 for i in range(nbrlrg)]
    reduced_bgs['class'] = [2 for i in range(nbrbg)]
    reduced_qsos['class'] = [3 for i in range(nbrqso)]

    if others_flag == 'all':
        all_objects = [reduced_others, reduced_lrgs, reduced_bgs, reduced_qsos]
    else:
        all_objects = [reduced_lrgs, reduced_bgs, reduced_qsos]

    magnitude_correction = 0.0
    all_objects_filled =  fill_empty_entries(all_objects, magnitude_correction)

    reduced_keys = list(reduced_lrgs.keys())
    nbr_features = len(reduced_keys)

    full_dataset = []

    for i in range(nbr_features):
        tmp_list = []
        for j in all_objects:
            tmp_list += list(j[reduced_keys[i]])
        full_dataset.append(tmp_list)

    full_table_dataset = np.transpose(np.array(full_dataset, dtype='float'))

    savepath = os.path.join(script_dir, 'datasets', fits_filename, others_flag + '-others_' + all_constraints_str + '-constraints')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataset_filename = 'BG_LRG_QSO_classification'

    all_fits_column = []
    for i in range(nbr_features):
        all_fits_column.append(fits.Column(name=reduced_keys[i], array=full_table_dataset[:,i], format='D'))
    dataset = fits.BinTableHDU.from_columns(all_fits_column)

    try:
        dataset.writeto(os.path.join(savepath, dataset_filename + '.fits'))
    except OSError as e:
        print(e)

    if others_flag == 'all':
        class_dict = {'nbr_objects': [nbrothers, nbrlrg, nbrbg, nbrqso], 'labels': [0, 1, 2, 3]}
    else:
        class_dict = {'nbr_objects': [nbrlrg, nbrbg, nbrqso], 'labels': [1, 2, 3]}

    generate_cross_validation_datasets(cv_ratios, class_dict, full_table_dataset, reduced_keys, dataset_filename, others_flag, all_constraints_str, fits_filename)

def binary_dataset(fits_path, fits_filename, others_flag, constraints, object_to_select):

    # Extract data from .fits file

    data, data_keys = read_fits(fits_path, fits_filename)
    fits_filename, _ = os.path.splitext(fits_filename)

    if constraints != 'no':
        data = apply_constraints(constraints, data)
        all_constraints_str = ''
        for s in list(constraints.keys()):
            if all_constraints_str:
                all_constraints_str += '+' + s
            else:
                all_constraints_str = s
    else:
        all_constraints_str = 'no'

    # Select objects of interest from data

    elgs, nbrelg = ELG_selector(data, data_keys)
    lrgs, nbrlrg = LRG_selector(data, data_keys)
    bgs, nbrbg = BG_selector(data, data_keys)
    qsos, nbrqso = QSO_selector(data, data_keys)
    others, nbrothers = Other_selector(data, data_keys)

    nbr_objects = nbrelg + nbrlrg + nbrbg + nbrqso + nbrothers

    # Define keys that we want to keep in dictionnaries
    if object_to_select == 'ELG':
        wanted_keys=['DES']
        exceptions_keys = ['DES_ra', 'DES_dec', 'DES_spread']
    else:
        wanted_keys = ['DES', 'VHS', 'WISE']
        exceptions_keys = ['isWISE', 'isVHS', 'VHS_class','DES_ra', 'DES_dec', 'DES_spread']

    # Delete non wanted keys from dictionnaries

    reduced_elgs = reduce_dict(elgs, wanted_keys, exceptions_keys)
    reduced_lrgs = reduce_dict(lrgs, wanted_keys, exceptions_keys)
    reduced_bgs = reduce_dict(bgs, wanted_keys, exceptions_keys)
    reduced_qsos = reduce_dict(qsos, wanted_keys, exceptions_keys)
    reduced_others = reduce_dict(others, wanted_keys, exceptions_keys)

    #  Define the class labels

    reduced_elgs, reduced_lrgs, reduced_bgs, reduced_qsos, reduced_others = labelize(reduced_elgs, nbrelg, reduced_lrgs, nbrlrg, reduced_bgs, nbrbg, reduced_qsos, nbrqso, reduced_others, nbrothers, object_to_select)

    #  Store all objects in a list for generalization of the rest of the function.
    #  Instead of using the full_data flag each time we want to access the data,
    #  we can now loop on the all_objects list which is itself defined depending on the
    #  full_data flag.

    if others_flag == 'all':
        all_objects = [ reduced_others, reduced_elgs, reduced_lrgs, reduced_bgs, reduced_qsos]
    else :
        all_objects = [reduced_elgs, reduced_lrgs, reduced_bgs, reduced_qsos]

    reduced_keys = list(reduced_elgs.keys())
    nbr_features = len(reduced_keys)

    #   Prepare the dataset using list concatenation and a list of list which is convenient to convert
    #   to a numpy array later.
    #   As an example :
    #   list_of_list = []
    #   [a,b] + [c,d] = [a,b,c,d]
    #   [e,f] + [g,h] = [e,f,g,h]
    #   list_of_list.append([a,b] + [c,d])
    #   list_of_list.append([a,b] + [c,d])
    #   print(list_of_list) -> [[a,b,c,d], [e,f,g,h]]
    #   numpy_array = np.array(list_of_list)
    #   print(numpy_array) -> [[a,b,c,d]
    #                           [e,f,g,h]]
    # Here, each row in the final array is an object and each column is the value of that object for the corresponding key contained in reduced keys
    # and respecting its order
    if object_to_select != 'ELG':
        magnitude_correction = 0.0
        all_objects_filled =  fill_empty_entries(all_objects, magnitude_correction)

    full_dataset = []

    for i in range(nbr_features):
        tmp_list = []
        for j in all_objects:
            tmp_list += list(j[reduced_keys[i]])
        full_dataset.append(tmp_list)

    full_table_dataset = np.transpose(np.array(full_dataset, dtype='float'))

    # Save the dataset in a .csv file with a path depending on the full_data flag

    savepath = os.path.join(script_dir, 'datasets', fits_filename, others_flag + '-others_' + all_constraints_str + '-constraints')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataset_filename = object_to_select + '_binary_classification'

    all_fits_column = []
    for i in range(nbr_features):
        all_fits_column.append(fits.Column(name=reduced_keys[i], array=full_table_dataset[:,i], format='D'))
    dataset = fits.BinTableHDU.from_columns(all_fits_column)
    try:
        dataset.writeto(os.path.join(savepath, dataset_filename + '.fits'))
    except OSError as e:
        print(e)

    # Define a class dictionnary containing the number of object for each class as well as the class label associated to those objects
    # As in python 3 dictionnary are ordered, it is important that the list in 'nbr_objects' key of the dictionnary and the list in 'labels'
    # key of the dictionnary follow the same order !

    nbr_zero_label = 0
    nbr_one_label = 0
    for i in all_objects:
        if i['class']:
            if i['class'][0] == 0:
                 nbr_zero_label += len(i['class'])
            elif i['class'][0] == 1:
                nbr_one_label += len(i['class'])
    print(nbr_one_label)
    class_dict = {'nbr_objects': [nbr_zero_label, nbr_one_label], 'labels': [0, 1]}

    # Generate cross validation dataset from the full dataset using the generate_cross_validation_datasets function

    generate_cross_validation_datasets(cv_ratios, class_dict, full_table_dataset, reduced_keys, dataset_filename, others_flag, all_constraints_str, fits_filename)
#
script_path = os.path.realpath(__file__)
script_dir, _ = os.path.split(script_path)
fits_filename = "4MOST.CatForGregoire.11Oct2018.zphot.fits"
cv_ratios = [[90, 10, 10], [80, 20, 5]]
others_flag = 'all'
constraints = 'no'
all_constraints_str = constraints_to_str(constraints)
#
# object_to_select = 'ELG'
# binary_dataset(script_dir, fits_filename, others_flag, constraints, object_to_select)

noelg_multiclass_dataset(script_dir, fits_filename, others_flag, constraints)

# generate_all_datasets()
