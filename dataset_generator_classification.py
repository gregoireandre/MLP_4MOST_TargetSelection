import os
import sys
import math
import warnings
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

from utils_classification import *

class Classification_dataset_generator():

    def __init__(self, fits_filename='4MOST.CatForGregoire.05Dec2018.zphot.fits', classification_problem='BG_LRG_ELG_QSO_classification', wanted_keys=['DES', 'VHS', 'WISE'], exceptions_keys=['isWISE', 'isVHS', 'VHS_class','DES_ra', 'DES_dec', 'DES_spread'], cv_ratios=[[80, 20, 20]], constraints='no', zphot_conf_threshold=0.0, zphot_risk_threshold=1.0, zphot_safe_threshold=0.0, zphot_estimator='ephor'):

        self.fits_filename = fits_filename
        self.classification_problem = classification_problem
        self.wanted_keys = wanted_keys
        self.exceptions_keys = exceptions_keys
        self.cv_ratios = cv_ratios
        self.constraints = constraints
        self.zphot_conf_threshold = zphot_conf_threshold
        self.zphot_risk_threshold = zphot_risk_threshold
        self.zphot_safe_threshold = zphot_safe_threshold
        self.zphot_key = 'HSC_' + zphot_estimator + '_zphot'
        self.zphotconf_key = 'HSC_' + zphot_estimator + '_zphotconf'
        self.zphotrisk_key = 'HSC_' + zphot_estimator + '_zphotrisk'

        self.exceptions_keys.append(self.zphotconf_key)
        self.exceptions_keys.append(self.zphotrisk_key)

        self.script_path = os.path.realpath(__file__)
        self.script_dir, _ = os.path.split(self.script_path)
        self.fits_filename_noext, _ = os.path.splitext(self.fits_filename)
        self.savepath = os.path.join(self.script_dir, 'datasets', self.fits_filename_noext, self.constraints + '-constraints')

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.data, self.data_keys = read_fits_as_dict(os.path.join(self.script_dir, self.fits_filename))

        self.reduced_keys = []
        for idx, i in enumerate(self.data_keys):
            if any(substring in i for substring in self.wanted_keys) and (i not in self.exceptions_keys):
                if i == 'DES_id':
                    self.reduced_keys.append(i)
                    self.DES_feature_idx = idx
                    continue
                elif 'id' in i:
                    continue
                elif ('noerr' in self.constraints) and ('err' in i):
                    continue
                else:
                    self.reduced_keys.append(i)

        print('Initialized')

        return

    def process_dataset(self):

        all_objects = self.object_selector()

        for idx, i in enumerate(all_objects):
            all_objects[idx] = self.reorder_dict(i)

        if 'colors' in self.constraints:
            for idx, i in enumerate(all_objects):
                all_objects[idx] = self.color_dict(i)

        self.data = None

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

        dataset = []
        for i in range(self.nbr_features):
            tmp_list = []
            for j in all_objects:
                tmp_list += list(j[self.reduced_keys[i]])
            dataset.append(tmp_list)

        del all_objects

        self.table_dataset = np.transpose(np.array(dataset, dtype='float'))
        del dataset

        return

    def generate_dataset(self):

        unique, counts = np.unique(self.table_dataset[:, -1], return_counts=True)
        self.class_count_dict = dict(zip(unique, counts))
        self.nbr_objects = np.sum(counts)
        print('Class count : ', self.class_count_dict)
        print('Remove multiple labels in favor of minority class')
        self.remove_multiple_entries()
        print('New Class count : ', self.class_count_dict)

        self.labels = list(self.class_count_dict.keys())
        self.labels_count = list(self.class_count_dict.values())
        self.nbr_classes = len(self.labels)

        # Save the dataset in a .fits file

        fits_column = []
        for i in range(self.nbr_features):
            if i == self.DES_feature_idx:
                fits_column.append(fits.Column(name=self.reduced_keys[i], array=self.table_dataset[:,i], format='J'))
            else:
                fits_column.append(fits.Column(name=self.reduced_keys[i], array=self.table_dataset[:,i], format='E'))
        dataset = fits.BinTableHDU.from_columns(fits_column)

        del fits_column

        try:
            dataset.writeto(os.path.join(self.savepath, self.classification_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '.fits'))
            print(self.classification_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '.fits generated')
        except OSError as e:
            print(e)

        del dataset

        # Generate cross validation dataset from the full dataset using the generate_cross_testing_datasets function

        self.generate_test_train_val_datasets()

    def object_selector(self):
        constraints_mask = self.compute_constraints_mask(self.constraints)

        x = self.data['VHS_j']-self.data['VHS_k']
        y = self.data['VHS_j']-self.data['WISE_w1']
        isbg = (
                    (self.data['isVHS']==True) &
                    (self.data['isWISE']==True) &
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
        isbgzphot = (
                       (self.data[self.zphot_key]>0.05) &
                       (self.data[self.zphot_key]<0.4)
                    )
        isbg = isbg & isbgzphot & ~constraints_mask
        BG_dict = {n:self.data[n][isbg] for n in self.reduced_keys}
        nbr_bg = len(self.data[isbg])
        BG_dict['confidence'] = self.compute_confidence_value(isbg)
        BG_dict['class'] = [1 for i in range(nbr_bg)]

        x = self.data['VHS_j']-self.data['VHS_k']
        y = self.data['VHS_j']-self.data['WISE_w1']
        islrg = (
                    (self.data['isVHS']==True) &
                    (self.data['isWISE']==True) &
                    (18<self.data['VHS_j']) &
                    (self.data['VHS_j']<19.5) &
                    (self.data['VHS_class']==1) &
                    (x>0.25) &
                    (x<1.50) &
                    (y<1.50) &
                    (y>1.6*x-1.5) &
                    (y>-0.5*x+0.65)
                )
        islrgzphot = (
                       (self.data[self.zphot_key]>0.4) &
                       (self.data[self.zphot_key]<0.75)
                     )
        islrg = islrg & islrgzphot & ~constraints_mask
        LRG_dict = {n:self.data[n][islrg] for n in self.reduced_keys}
        nbr_lrg = len(self.data[islrg])
        LRG_dict['confidence'] = self.compute_confidence_value(islrg)
        LRG_dict['class'] = [2 for i in range(nbr_lrg)]

        x = self.data['DES_g']-self.data['DES_r']
        y = self.data['DES_r']-self.data['DES_i']
        iselg = (
                    (21<self.data['DES_g']) &
                    (self.data['DES_g']<23.2) &
                    (0.5-2.5*x<y) &
                    (y<3.5-2.5*x) &
                    (0.4*x+0.3<y) &
                    (y<0.4*x+0.9)
                )

        iselgzphot = (
                        (self.data[self.zphot_key]>0.75) &
                        (self.data[self.zphot_key]<1.1)
                     )
        iselg = iselg & iselgzphot & ~constraints_mask
        ELG_dict = {n:self.data[n][iselg] for n in self.reduced_keys}
        nbr_elg = len(self.data[iselg])
        ELG_dict['confidence'] = self.compute_confidence_value(iselg)
        ELG_dict['class'] = [3 for i in range(nbr_elg)]

        isqso = (
                    (self.data['isWISE']==True) &
                    (self.data['isDR14QSO']==True) &
                    (self.data['DR14QSO_zspec']>1.1)
                )

        if '+zsafe' in self.constraints:
            constraints_qso = self.constraints.replace("+zsafe", "")
            constraints_mask_qso = self.compute_constraints_mask(constraints_qso)
        else:
            constraints_mask_qso = constraints_mask

        isqso = isqso & ~constraints_mask_qso

        QSO_dict = {n:self.data[n][isqso] for n in self.reduced_keys}
        nbr_qso = len(self.data[isqso])
        QSO_dict['confidence'] = [1.0]*nbr_qso
        QSO_dict['class'] = [4 for i in range(nbr_qso)]
        # if self.zphotconf_key in self.reduced_keys:
        #     for i in range(nbr_qso):
        #         QSO_dict[self.zphotconf_key][i] = 1.0

        isothers = ~isbg*~islrg*~iselg*~isqso*~constraints_mask
        Others_dict = {n:self.data[n][isothers] for n in self.reduced_keys}
        nbr_other = len(self.data[isothers])
        Others_dict['confidence'] = self.compute_confidence_value(isothers)
        Others_dict['class'] = [0 for i in range(nbr_other)]

        #  Store all objects in a list for generalization of the rest of the function.

        all_objects = [Others_dict, BG_dict, LRG_dict, ELG_dict, QSO_dict]

        return all_objects

    def compute_constraints_mask(self, constraints):

        constraints_mask = np.zeros(self.data['DES_g'].shape[0], dtype=bool)
        constraints_list = constraints.split('+')

        for i in constraints_list:
            if i == 'noout':
                isoutlier = (
                                (self.data['DES_g']>=90) | (self.data['DES_r']>=90) | (self.data['DES_i']>=90) | (self.data['DES_z']>=90) | (self.data['DES_y']>=90) | (self.data['VHS_j']>=90) | (self.data['VHS_k']>=90) | (self.data['WISE_w1']>=90) | (abs(self.data['DES_spread'])>1) |
                                (self.data['DES_gerr']>=1) | (self.data['DES_rerr']>=1) | (self.data['DES_ierr']>=1) | (self.data['DES_zerr']>=1) | (self.data['DES_yerr']>=1) | (self.data['VHS_jerr']>=1) | (self.data['VHS_kerr']>=1) | (self.data['WISE_w1err']>=1)
                            )
                constraints_mask += isoutlier
            elif i == 'zsafe':
                # isnotzsafe = (
                #                (self.data[self.zphotconf_key] < self.zphot_conf_threshold) |
                #                (self.data[self.zphotrisk_key] > self.zphot_risk_threshold)
                #              )
                isnotzsafe = (
                               (self.data[self.zphotconf_key]*(1-self.data[self.zphotrisk_key]) < self.zphot_safe_threshold)
                             )
                constraints_mask += isnotzsafe
            elif i == 'nostar':
                isstar = (
                            (self.data['VVDS_zspec'] == 0.0) |
                            (self.data['DR14QSO_zspec'] == 0.0)
                         )
                constraints_mask += isstar
            elif i == 'bright':
                isbright = (
                           ((self.data['isWISE']==True) & (self.data['isDR14QSO']==True)) |
                           ((self.data['VHS_j']>16) & (self.data['VHS_j']<19.5)) |
                           ((self.data['DES_g']>21) & (self.data['DES_g']<23.2))
                           )
                constraints_mask += ~isbright
            else:
                print('No constraints defined for "' + i + '" tag')
                print('Continue without applying "' + i + '" constraint')
                continue

        return constraints_mask

    def compute_confidence_value(self, mask):

        confidence_values = [(1-risk)*conf for risk,conf in zip(self.data[self.zphotrisk_key][mask], self.data[self.zphotconf_key][mask])]

        interpolator_conf = interp1d([np.amin(confidence_values),1.0],[0.0,1.0])

        for idx, i in enumerate(confidence_values):
            confidence_values[idx] = interpolator_conf(i)

        return confidence_values

    def reorder_dict(self, object_dict):

        keys_order = ['DES_id', self.zphot_key,'confidence', 'DES_g', 'DES_r', 'DES_i', 'DES_z', 'DES_y', 'VHS_j', 'VHS_k', 'WISE_w1', 'DES_gerr', 'DES_rerr', 'DES_ierr', 'DES_zerr', 'DES_yerr', 'VHS_jerr', 'VHS_kerr', 'WISE_w1err', 'DES_spread', 'class']
        ordered_dict = {}

        for i in keys_order:
            ordered_dict[i] = object_dict[i]

        self.reduced_keys = list(ordered_dict.keys())
        self.nbr_features = len(self.reduced_keys)

        return ordered_dict

    def color_dict(self, object_dict):

        object_dict_color = {}

        object_dict_color['DES_id'] = object_dict['DES_id']
        object_dict_color[self.zphotconf_key] = object_dict[self.zphotconf_key]
        object_dict_color['DES_g-DES_r'] = object_dict['DES_g'] - object_dict['DES_r']
        object_dict_color['DES_r-DES_i'] = object_dict['DES_r'] - object_dict['DES_i']
        object_dict_color['DES_i-DES_z'] = object_dict['DES_i'] - object_dict['DES_z']
        object_dict_color['DES_z-DES_y'] = object_dict['DES_z'] - object_dict['DES_y']
        object_dict_color['VHS_j-VHS_k'] = object_dict['VHS_j'] - object_dict['VHS_k']
        object_dict_color['VHS_j-WISE_w1'] = object_dict['VHS_j'] - object_dict['WISE_w1']
        if 'mag' in self.constraints:
            object_dict_color['DES_g'] = object_dict['DES_g']
            object_dict_color['DES_r'] = object_dict['DES_r']
            object_dict_color['DES_i'] = object_dict['DES_i']
            object_dict_color['DES_z'] = object_dict['DES_z']
            object_dict_color['DES_y'] = object_dict['DES_y']
            object_dict_color['VHS_j'] = object_dict['VHS_j']
            object_dict_color['VHS_k'] = object_dict['VHS_k']
            object_dict_color['WISE_w1'] = object_dict['WISE_w1']
        if 'noerr' not in self.constraints:
            object_dict_color['DES_gerr'] = object_dict['DES_gerr']
            object_dict_color['DES_rerr'] = object_dict['DES_rerr']
            object_dict_color['DES_ierr'] = object_dict['DES_ierr']
            object_dict_color['DES_zerr'] = object_dict['DES_zerr']
            object_dict_color['DES_yerr'] = object_dict['DES_yerr']
            object_dict_color['VHS_jerr'] = object_dict['VHS_jerr']
            object_dict_color['VHS_kerr'] = object_dict['VHS_kerr']
            object_dict_color['WISE_w1err'] = object_dict['WISE_w1err']
        if 'DES_spread' not in self.exceptions_keys:
            object_dict_color['DES_spread'] = object_dict['DES_spread']
        object_dict_color['class'] = object_dict['class']

        self.reduced_keys = list(object_dict_color.keys())
        self.nbr_features = len(self.reduced_keys)

        return object_dict_color

    def remove_multiple_entries(self):

        # Get the number of count and associated DES_id for the whole dataset
        DES_id_array, count_array = np.unique(self.table_dataset[:, 0], return_counts=True)

        # Get the index of DES_ids for which there is more than one object (eg "multiple entries")
        idx_multiple_entries = np.where(count_array > 1)[0]
        if len(idx_multiple_entries) == 0:
            print('No multiple entries')
            return
        # Get the corresponding DES_ids for which there is more than one object
        DES_id_multiple_entries = DES_id_array[idx_multiple_entries]

        indexes_to_keep = list(range(0, self.nbr_objects))
        indexes_to_delete = []

        test = np.zeros((len(DES_id_multiple_entries), np.amax(count_array)))

        for idxi, i in enumerate(DES_id_multiple_entries):
            # Get indexes of multiple entries in the dataset table
            table_dataset_idx = np.where(self.table_dataset[:, 0]==i)[0]
            # Get class corresponding to those multiple entries
            multiple_entries_classes = self.table_dataset[table_dataset_idx, -1]
            for idxk, k in enumerate(multiple_entries_classes):
                test[idxi, idxk] = k
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

        test_unique, test_count = np.unique(test, axis=0, return_counts=True)
        print('Found ', sum(test_count), ' objects with multiple labels')
        for j in range(len(test_unique)):
            print(test_count[j], ' objects have following multi-labels : ', test_unique[j])

        for i in sorted(indexes_to_delete, reverse=True):
            del indexes_to_keep[i]

        self.table_dataset = self.table_dataset[indexes_to_keep,:]
        self.nbr_objects = self.table_dataset.shape[0]

        return

    def generate_test_train_val_datasets(self):

        isclass = {}
        for label in self.labels:
            isclass[label] = (self.table_dataset[:,-1]==label)
        for i in self.cv_ratios:
            train_ratio = i[0]/100
            test_ratio = i[1]/100
            mask_train = np.ones(self.nbr_objects,dtype=bool)
            table_dataset_train = []
            idx_test = []
            nbr_object_test = []
            for h in self.labels_count:
                nbr_object_test.append(math.floor(h*test_ratio))
            for labidx, label in enumerate(self.labels):
                avail = np.where(isclass[label])[0]
                idx_test += avail[:nbr_object_test[labidx]].tolist()
                mask_train[avail[:nbr_object_test[labidx]]] = False
            idx_train = np.where(mask_train)[0]
            table_dataset_test = self.table_dataset[idx_test, :]
            table_dataset_train = self.table_dataset[idx_train,:]
            self.generate_cross_validation_datasets(table_dataset_train)
            fits_column_test = []
            for g in range(self.nbr_features):
                if g == self.DES_feature_idx:
                    fits_column_test.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_test[:,g], format='J'))
                else:
                    fits_column_test.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_test[:,g], format='E'))
            test = fits.BinTableHDU.from_columns(fits_column_test)
            try:
                test.writeto(os.path.join(self.savepath, self.classification_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_test_' + str(i[0]) + '_' + str(i[1]) + '.fits'))
                print(self.classification_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_test_' + str(i[0]) + '_' + str(i[1]) + '.fits generated')
            except OSError as e:
                print(e)

        return

    def generate_cross_validation_datasets(self, table_dataset):

        unique, counts = np.unique(table_dataset[:, -1], return_counts=True)
        class_count_dict_train = dict(zip(unique, counts))
        labels_count_train = list(class_count_dict_train.values())
        nbr_objects_train = table_dataset.shape[0]

        isclass = {}
        for label in self.labels:
            isclass[label] = (table_dataset[:,-1]==label)

        for i in self.cv_ratios:
            train_ratio = i[0]/100
            val_ratio = (i[2]/100)/train_ratio
            nbr_fold = 1.0/val_ratio
            if ((nbr_fold % 1) != 0):
                print('Error : The validation ratio should be a divider of 100')
                print('Exit without cross validation dataset generation')
                return
            indexes_chosen_val = np.zeros(nbr_objects_train,dtype=bool)
            for j in range(int(nbr_fold)):
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
                    if g == self.DES_feature_idx:
                        fits_column_train.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_train[:,g], format='J'))
                        fits_column_val.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_val[:,g], format='J'))
                    else:
                        fits_column_train.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_train[:,g], format='E'))
                        fits_column_val.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_val[:,g], format='E'))
                val = fits.BinTableHDU.from_columns(fits_column_val)
                train = fits.BinTableHDU.from_columns(fits_column_train)
                try:
                    val.writeto(os.path.join(self.savepath, self.classification_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_val_' + str(i[0]) + '_' + str(i[2]) +  '_' + str(j+1) + '.fits'))
                    print(self.classification_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_val_' + str(i[0]) + '_' + str(i[2]) + '_' + str(j+1) + '.fits generated')
                except OSError as e:
                    print(e)
                try:
                    train.writeto(os.path.join(self.savepath, self.classification_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_train_' + str(i[0]) + '_' + str(i[2]) + '_' + str(j+1) + '.fits'))
                    print(self.classification_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_train_' + str(i[0]) + '_' + str(i[2]) + '_' + str(j+1) + '.fits generated')
                except OSError as e:
                    print(e)

        return

    def labelize(self):

        self.reduced_others['class'] = [0 for i in range(self.nbrothers)]
        self.reduced_bgs['class'] = [1 for i in range(self.nbrbg)]
        self.reduced_lrgs['class'] = [2 for i in range(self.nbrlrg)]
        self.reduced_elgs['class'] = [3 for i in range(self.nbrelg)]
        self.reduced_qsos['class'] = [4 for i in range(self.nbrqso)]

        return

if __name__ == '__main__':

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    classification_dataset_gen = Classification_dataset_generator(wanted_keys=['DES', 'VHS', 'WISE', 'HSC'], classification_problem='BG_LRG_ELG_QSO_classification_ephor', exceptions_keys=['isWISE', 'isVHS', 'DES_ra', 'DES_dec', 'VHS_class'], constraints='noout+bright', zphot_estimator='ephor')
    classification_dataset_gen.process_dataset()
    classification_dataset_gen.generate_dataset()

    classification_dataset_gen = Classification_dataset_generator(wanted_keys=['DES', 'VHS', 'WISE', 'HSC'], classification_problem='BG_LRG_ELG_QSO_classification_ephor', exceptions_keys=['isWISE', 'isVHS', 'DES_ra', 'DES_dec', 'VHS_class'], constraints='noout+bright+nostar', zphot_estimator='ephor')
    classification_dataset_gen.process_dataset()
    classification_dataset_gen.generate_dataset()

    classification_dataset_gen = Classification_dataset_generator(wanted_keys=['DES', 'VHS', 'WISE', 'HSC'], classification_problem='BG_LRG_ELG_QSO_classification_ephor', exceptions_keys=['isWISE', 'isVHS', 'DES_ra', 'DES_dec', 'VHS_class'], constraints='noout+bright+nostar+zsafe', zphot_safe_threshold=0.3, zphot_estimator='ephor')
    classification_dataset_gen.process_dataset()
    classification_dataset_gen.generate_dataset()
