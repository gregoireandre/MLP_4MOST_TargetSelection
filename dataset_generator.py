import os
import math
import time
import operator
import itertools
import numpy as np
from astropy.io import fits

from utils import *

class Classification_dataset_generator():

    def __init__(self, fits_filename='4MOST.CatForGregoire.11Oct2018.zphot.fits', classification_problem='BG_ELG_LRG_QSO_classification', data_imputation=0.0, wanted_keys=['DES', 'VHS', 'WISE'], exceptions_keys=['isWISE', 'isVHS', 'VHS_class','DES_ra', 'DES_dec', 'DES_spread'], cv_ratios=[[80, 20, 5]], others_flag='all', constraints='no'):

        self.fits_filename = fits_filename
        self.classification_problem = classification_problem
        self.data_imputation = data_imputation
        self.wanted_keys = wanted_keys
        self.exceptions_keys = exceptions_keys
        self.cv_ratios = cv_ratios
        self.others_flag = others_flag
        self.constraints = constraints

        self.script_path = os.path.realpath(__file__)
        self.script_dir, _ = os.path.split(self.script_path)
        self.all_constraints_str = constraints_to_str(self.constraints)
        self.fits_filename_noext, _ = os.path.splitext(self.fits_filename)
        self.savepath = os.path.join(self.script_dir, 'datasets', self.fits_filename_noext, self.others_flag + '-others_' + self.all_constraints_str + '-constraints_' + str(self.data_imputation) + '-imputation')

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.data, self.data_keys = read_fits_as_dict(os.path.join(self.script_dir, self.fits_filename))

        if self.constraints != 'no':
            self.data = apply_constraints()


        return

    def generate_dataset(self):

        # Select objects of interest from data

        elgs, self.nbrelg = self.ELG_selector()
        lrgs, self.nbrlrg = self.LRG_selector()
        bgs, self.nbrbg = self.BG_selector()
        qsos, self.nbrqso = self.QSO_selector()
        others, self.nbrothers = self.Other_selector()

        # Delete non wanted keys from dictionnaries

        self.reduced_others = self.reduce_dict(others)
        self.reduced_elgs = self.reduce_dict(elgs)
        self.reduced_lrgs = self.reduce_dict(lrgs)
        self.reduced_bgs = self.reduce_dict(bgs)
        self.reduced_qsos = self.reduce_dict(qsos)

        #  Define the class labels
        self.labelize()

        self.reduced_keys = list(self.reduced_elgs.keys())
        self.nbr_features = len(self.reduced_keys)

        #  Store all objects in a list for generalization of the rest of the function.
        #  Instead of using the full_data flag each time we want to access the data,
        #  we can now loop on the all_objects list which is itself defined depending on the
        #  full_data flag.

        if self.others_flag == 'all':
            all_objects = [self.reduced_others, self.reduced_elgs, self.reduced_lrgs, self.reduced_bgs, self.reduced_qsos]
        else:
            all_objects = [self.reduced_elgs, self.reduced_lrgs, self.reduced_bgs, self.reduced_qsos]

        all_objects_filled =  self.fill_empty_entries(all_objects)

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

        self.dataset = []

        for i in range(self.nbr_features):
            tmp_list = []
            for j in all_objects_filled:
                tmp_list += list(j[self.reduced_keys[i]])
            self.dataset.append(tmp_list)

        self.table_dataset = np.transpose(np.array(self.dataset, dtype='float'))

        unique, counts = np.unique(self.table_dataset[:, -1], return_counts=True)
        self.class_count_dict = dict(zip(unique, counts))
        self.nbr_objects = np.sum(counts)

        self.remove_multiple_entries()

        self.labels = list(self.class_count_dict.keys())
        self.labels_count = list(self.class_count_dict.values())
        self.nbr_classes = len(self.labels)

        # Save the dataset in a .fits file

        fits_column = []
        for i in range(self.nbr_features):
            fits_column.append(fits.Column(name=self.reduced_keys[i], array=self.table_dataset[:,i], format='D'))
        self.dataset = fits.BinTableHDU.from_columns(fits_column)

        try:
            self.dataset.writeto(os.path.join(self.savepath, self.classification_problem + '.fits'))
        except OSError as e:
            print(e)

        # Generate cross validation dataset from the full dataset using the generate_cross_testing_datasets function

        self.generate_cross_testing_datasets()

    def apply_constraints(self):

        ops = { "<": operator.lt, ">": operator.gt, "==": operator.eq}

        for i in list(self.constraints.keys()):
            for j in self.constraints[i]:
                if 'isinconstraint' in locals():
                    isinconstraint &= ops[j[1]](float(j[0]), data[j[2]])
                else:
                    isinconstraint = ops[j[1]](float(j[0]), data[j[2]])
        constrained_data = self.data[isinconstraint]
        return constrained_data

    def ELG_selector(self):

        ####role####
        # Select ELGs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

        ####inputs####
        # data : Should be the data object outputed by the read_fits function which contains all the data
        # from the .fits file.
        # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

        ####outputs####
        # ELG_dict : A python dictionnary containing all the ELG's data ordered within the same keys as in the data object taken as input.
        # nbr_elg : The number of ELG that were selected

        x = self.data['DES_g']-self.data['DES_r']
        y = self.data['DES_r']-self.data['DES_i']
        iselg = (
                (self.data['HSC_zphot']>0.7) &
                (self.data['HSC_zphot']<1.1) &
                (21<self.data['DES_g']) &
                (self.data['DES_g']<23.2) &
                (0.5-2.5*x<y) &
                (y<3.5-2.5*x) &
                (0.4*x+0.3<y) &
                (y<0.4*x+0.9)
                )
        ELG_dict = {n:self.data[n][iselg] for n in self.data_keys}
        nbr_elg = len(self.data[iselg])
        return ELG_dict, nbr_elg

    def LRG_selector(self):

        ####role####
        # Select LRGs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

        ####inputs####
        # data : Should be the data object outputed by the read_fits function which contains all the data
        # from the .fits file.
        # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

        ####outputs####
        # LRG_dict : A python dictionnary containing all the LRG's data ordered within the same keys as in the data object taken as input.
        # nbr_lrg : The number of LRG that were selected

        x = self.data['VHS_j']-self.data['VHS_k']
        y = self.data['VHS_j']-self.data['WISE_w1']
        islrg = (
                (self.data['isVHS']==True) &
                (self.data['isWISE']==True) &
                (self.data['HSC_zphot']>0.4) &
                (self.data['HSC_zphot']<0.8) &
                (18<self.data['VHS_j']) &
                (self.data['VHS_j']<19.5) &
                (self.data['VHS_class']==1) &
                (x>0.25) &
                (x<1.50) &
                (y<1.50) &
                (y>1.6*x-1.5) &
                (y>-0.5*x+0.65)
                )
        LRG_dict = {n:self.data[n][islrg] for n in self.data_keys}
        nbr_lrg = len(self.data[islrg])
        return LRG_dict, nbr_lrg

    def BG_selector(self):

        ####role####
        # Select BGs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

        ####inputs####
        # data : Should be the data object outputed by the read_fits function which contains all the data
        # from the .fits file.
        # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

        ####outputs####
        # BG_dict : A python dictionnary containing all the bg's data ordered within the same keys as in the data object taken as input.
        # nbr_bg : The number of BG that were selected

        x = self.data['VHS_j']-self.data['VHS_k']
        y = self.data['VHS_j']-self.data['WISE_w1']
        isbg = (
               (self.data['isVHS']==True) &
               (self.data['isWISE']==True) &
               (self.data['HSC_zphot']>0.05) &
               (self.data['HSC_zphot']<0.4) &
               (16<self.data['VHS_j']) &
               (self.data['VHS_j']<18) &
               (self.data['VHS_class']==1) &
               (x>0.10) &
               (x<1.00) &
               (y>1.6*x-1.6) &
               (y<1.6*x-0.5) &
               (y>-0.5*x-1.0) &
               (y<-0.5*x+0.1)
               )
        BG_dict = {n:self.data[n][isbg] for n in self.data_keys}
        nbr_bg = len(self.data[isbg])
        return BG_dict, nbr_bg

    def QSO_selector(self):

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
                (self.data['isWISE']==True) &
                (self.data['isDR14QSO']==True)
                )
        QSO_dict = {n:self.data[n][isqso] for n in self.data_keys}
        nbr_qso = len(self.data[isqso])
        return QSO_dict, nbr_qso

    def Other_selector(self):

        ####role####
        # Select all entry that do not correspond to ELG, LRG, BG, QSO (eg "other") from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

        ####inputs####
        # data : Should be the data object outputed by the read_fits function which contains all the data
        # from the .fits file.
        # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

        ####outputs####
        # Others_dict : A python dictionnary containing all the other's data ordered within the same keys as in the data object taken as input.
        # nbr_other : The number of "other" that were selected

        x = self.data['DES_g']-self.data['DES_r']
        y = self.data['DES_r']-self.data['DES_i']
        iselg = (
                (self.data['HSC_zphot']>0.7) &
                (self.data['HSC_zphot']<1.1) &
                (21<self.data['DES_g']) &
                (self.data['DES_g']<23.2) &
                (0.5-2.5*x<y) &
                (y<3.5-2.5*x) &
                (0.4*x+0.3<y) &
                (y<0.4*x+0.9)
                )
        notelg = [not i for i in iselg]
        x = self.data['VHS_j']-self.data['VHS_k']
        y = self.data['VHS_j']-self.data['WISE_w1']
        islrg = (
                (self.data['isVHS']==True) &
                (self.data['isWISE']==True) &
                (self.data['HSC_zphot']>0.4) &
                (self.data['HSC_zphot']<0.8) &
                (18<self.data['VHS_j']) &
                (self.data['VHS_j']<19.5) &
                (self.data['VHS_class']==1) &
                (x>0.25) &
                (x<1.50) &
                (y<1.50) &
                (y>1.6*x-1.5) &
                (y>-0.5*x+0.65)
                )
        notlrg = [not i for i in islrg]
        x = self.data['VHS_j']-self.data['VHS_k']
        y = self.data['VHS_j']-self.data['WISE_w1']
        isbg = (
               (self.data['isVHS']==True) &
               (self.data['isWISE']==True) &
               (self.data['HSC_zphot']>0.05) &
               (self.data['HSC_zphot']<0.4) &
               (16<self.data['VHS_j']) &
               (self.data['VHS_j']<18) &
               (self.data['VHS_class']==1) &
               (x>0.10) &
               (x<1.00) &
               (y>1.6*x-1.6) &
               (y<1.6*x-0.5) &
               (y>-0.5*x-1.0) &
               (y<-0.5*x+0.1)
               )
        notbg = [not i for i in isbg]
        isqso = (
                (self.data['isWISE']==True) &
                (self.data['isDR14QSO']==True)
                )
        notqso = [not i for i in isqso]
        isothers = []
        for i in range(len(isqso)):
            isothers.append(not any([iselg[i], isbg[i], islrg[i], isqso[i]]))
        Others_dict = {n:self.data[n][isothers] for n in self.data_keys}
        nbr_other = len(self.data[isothers])
        return Others_dict, nbr_other

    def reduce_dict(self, data_dict):

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
            if not any(substring in i for substring in self.wanted_keys):
                del data_dict[i]
                continue
            if 'DES_id' in i:
                continue
            # ID's are not relevant informations so we delete it
            if 'id' in i:
                del data_dict[i]
                continue
            if i in self.exceptions_keys:
                del data_dict[i]
                continue

        return data_dict

    def fill_empty_entries(self, object_dict_list):

        ####role####
        # Fill empty entries in the dataset. In our case, there are empty entries for the

        ####inputs####
        # data : Should be the data object outputed by the read_fits function which contains all the data
        # from the .fits file.
        # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

        ####outputs####
        # QSO_dict : A python dictionnary containing all the QSO's data ordered within the same keys as in the data object taken as input.
        # nbr_qso : The number of QSO that were selected

        for i in object_dict_list:
            for j in self.reduced_keys:
                for idx, k in enumerate(i[j]):
                    if np.isnan(k):
                        i[j][idx] = self.data_imputation

        return object_dict_list

    def remove_multiple_entries(self):

        # Get the number of count and associated DES_id for the whole dataset
        DES_id_array, count_array = np.unique(self.table_dataset[:, 0], return_counts=True)

        # Get the index of DES_ids for which there is more than one object (eg "multiple entries")
        idx_multiple_entries = np.where(count_array > 1)[0]
        # Get the corresponding DES_ids for which there is more than one object
        DES_id_multiple_entries = DES_id_array[idx_multiple_entries]

        indexes_to_keep = list(range(0, self.nbr_objects))
        indexes_to_delete = []

        for i in DES_id_multiple_entries:
            # Get indexes of multiple entries in the dataset table
            table_dataset_idx = np.where(self.table_dataset[:, 0]==i)[0]
            # Get class corresponding to those multiple entries
            multiple_entries_classes = self.table_dataset[table_dataset_idx, -1]
            # Get the number of elements for each class referenced in multiple_entries_classes
            multiple_entries_count = [self.class_count_dict[x] for x in multiple_entries_classes]
            # We want to keep this entry for the class that has the less elements
            min_count_idx = multiple_entries_count.index(min(multiple_entries_count))

            # Among the multiple entries, we want to keep the object which is associated to the minority class
            # The index of the latter in the dataset table is the following table_dataset_idx[min_count_idx]
            for idx, j in enumerate(table_dataset_idx):
                if idx != min_count_idx:
                    indexes_to_delete.append(table_dataset_idx[idx])
                    self.class_count_dict[multiple_entries_classes[idx]] -= 1

        for i in sorted(indexes_to_delete, reverse=True):
            del indexes_to_keep[i]

        self.table_dataset = self.table_dataset[indexes_to_keep,:]
        self.nbr_objects = self.table_dataset.shape[0]

        return

    def generate_cross_testing_datasets(self):

        isclass = {}
        for label in self.labels:
            isclass[label] = (self.table_dataset[:,-1]==label)

        for i in self.cv_ratios:
            train_ratio = i[0]/100
            test_ratio = i[1]/100
            dataset_idx = i[2]
            indexes_chosen_test = np.zeros(self.nbr_objects,dtype=bool)
            for j in range(dataset_idx):
                mask_train = np.ones(self.nbr_objects,dtype=bool)
                table_dataset_train = []
                idx_test = []
                nbr_object_test = []
                for h in self.labels_count:
                    nbr_object_test.append(math.floor(h*test_ratio))
                for labidx, label in enumerate(self.labels):
                    avail = np.where((isclass[label]) &  (~indexes_chosen_test))[0]
                    idx_test += avail[:nbr_object_test[labidx]].tolist()
                    indexes_chosen_test[avail[:nbr_object_test[labidx]]] = True
                    mask_train[avail[:nbr_object_test[labidx]]] = False
                idx_train = np.where(mask_train)[0]
                table_dataset_test = self.table_dataset[idx_test, :]
                table_dataset_train = self.table_dataset[idx_train,:]
                self.generate_cross_validation_datasets(table_dataset_train, j+1)
                fits_column_test = []
                for g in range(self.nbr_features):
                    fits_column_test.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_test[:,g], format='D'))
                test = fits.BinTableHDU.from_columns(fits_column_test)
                try:
                    test.writeto(os.path.join(self.savepath, self.classification_problem + '_test_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits'))
                    print(self.classification_problem + '_test_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits')
                except OSError as e:
                    print(e)

        return

    def generate_cross_validation_datasets(self, table_dataset, dataset_idx):

        unique, counts = np.unique(table_dataset[:, -1], return_counts=True)
        class_count_dict_train = dict(zip(unique, counts))
        labels_count_train = list(class_count_dict_train.values())
        nbr_objects_train = table_dataset.shape[0]

        isclass = {}
        for label in self.labels:
            isclass[label] = (table_dataset[:,-1]==label)

        for i in self.cv_ratios:
            train_ratio = i[0]/100
            val_ratio = i[1]/100
            nbr_fold = i[2]
            indexes_chosen_val = np.zeros(nbr_objects_train,dtype=bool)
            for j in range(nbr_fold):
                mask_train = np.ones(nbr_objects_train,dtype=bool)
                table_dataset_train = []
                idx_val = []
                nbr_object_val = []
                for h in labels_count_train:
                    nbr_object_val.append(math.floor(h*val_ratio))
                for labidx, label in enumerate(self.labels):
                    avail = np.where((isclass[label]) &  (~indexes_chosen_val))[0]
                    idx_val += avail[:nbr_object_val[labidx]].tolist()
                    indexes_chosen_val[avail[:nbr_object_val[labidx]]] = True
                    mask_train[avail[:nbr_object_val[labidx]]] = False
                idx_train = np.where(mask_train)[0]
                table_dataset_val = table_dataset[idx_val, :]
                table_dataset_train= table_dataset[idx_train,:]
                fits_column_val = []
                fits_column_train = []
                for g in range(self.nbr_features):
                    fits_column_train.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_train[:,g], format='D'))
                    fits_column_val.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_val[:,g], format='D'))
                val = fits.BinTableHDU.from_columns(fits_column_val)
                train = fits.BinTableHDU.from_columns(fits_column_train)
                try:
                    val.writeto(os.path.join(self.savepath, self.classification_problem + '_val_' + str(i[0]) + '_' + str(i[1]) + '_' + str(dataset_idx) + '_' + str(j+1) + '.fits'))
                    print(self.classification_problem + '_val_' + str(i[0]) + '_' + str(i[1]) + '_' + str(dataset_idx) + '_' + str(j+1) + '.fits')
                except OSError as e:
                    print(e)
                try:
                    train.writeto(os.path.join(self.savepath, self.classification_problem+ '_train_' + str(i[0]) + '_' + str(i[1]) + '_' + str(dataset_idx) + '_' + str(j+1) + '.fits'))
                    print(self.classification_problem+ '_train_' + str(i[0]) + '_' + str(i[1]) + '_' + str(dataset_idx) + '_' + str(j+1) + '.fits')
                except OSError as e:
                    print(e)

        return

    def labelize(self):

        if 'binary' in self.classification_problem:

            if 'ELG' in self.classification_problem:

                self.reduced_elgs['class'] = [1 for i in range(self.nbrelg)]
                self.reduced_lrgs['class'] = [0 for i in range(self.nbrlrg)]
                self.reduced_bgs['class'] = [0 for i in range(self.nbrbg)]
                self.reduced_qsos['class'] = [0 for i in range(self.nbrqso)]
                self.reduced_others['class'] = [0 for i in range(self.nbrothers)]

            elif 'LRG' in self.classification_problem:

                self.reduced_elgs['class'] = [0 for i in range(self.nbrelg)]
                self.reduced_lrgs['class'] = [1 for i in range(self.nbrlrg)]
                self.reduced_bgs['class'] = [0 for i in range(self.nbrbg)]
                self.reduced_qsos['class'] = [0 for i in range(self.nbrqso)]
                self.reduced_others['class'] = [0 for i in range(self.nbrothers)]

            elif 'BG' in self.classification_problem:

                self.reduced_elgs['class'] = [0 for i in range(self.nbrelg)]
                self.reduced_lrgs['class'] = [0 for i in range(self.nbrlrg)]
                self.reduced_bgs['class'] = [1 for i in range(self.nbrbg)]
                self.reduced_qsos['class'] = [0 for i in range(self.nbrqso)]
                self.reduced_others['class'] = [0 for i in range(self.nbrothers)]

            elif 'QSO' in self.classification_problem:

                self.reduced_elgs['class'] = [0 for i in range(self.nbrelg)]
                self.reduced_lrgs['class'] = [0 for i in range(self.nbrlrg)]
                self.reduced_bgs['class'] = [0 for i in range(self.nbrbg)]
                self.reduced_qsos['class'] = [1 for i in range(self.nbrqso)]
                self.reduced_others['class'] = [0 for i in range(self.nbrothers)]

            elif 'Others' in self.classification_problem:

                self.reduced_elgs['class'] = [1 for i in range(self.nbrelg)]
                self.reduced_lrgs['class'] = [1 for i in range(self.nbrlrg)]
                self.reduced_bgs['class'] = [1 for i in range(self.nbrbg)]
                self.reduced_qsos['class'] = [1 for i in range(self.nbrqso)]
                self.reduced_others['class'] = [0 for i in range(self.nbrothers)]

        else:

            if 'QSO' in self.classification_problem:

                self.reduced_others['class'] = [0 for i in range(self.nbrothers)]
                self.reduced_elgs['class'] = [1 for i in range(self.nbrelg)]
                self.reduced_lrgs['class'] = [2 for i in range(self.nbrlrg)]
                self.reduced_bgs['class'] = [3 for i in range(self.nbrbg)]
                self.reduced_qsos['class'] = [4 for i in range(self.nbrqso)]

            else:

                self.reduced_others['class'] = [0 for i in range(self.nbrothers)]
                self.reduced_elgs['class'] = [1 for i in range(self.nbrelg)]
                self.reduced_lrgs['class'] = [2 for i in range(self.nbrlrg)]
                self.reduced_bgs['class'] = [3 for i in range(self.nbrbg)]
                self.reduced_qsos['class'] = [0 for i in range(self.nbrqso)]

        return

classification_dataset_gen = Classification_dataset_generator()
classification_dataset_gen.generate_dataset()
