import os
import sys
import math
import warnings
import numpy as np
from astropy.io import fits

from utils_regression import *

class Regression_dataset_generator():

    def __init__(self, fits_filename='4MOST.CatForGregoire.05Dec2018.zphot.fits', regression_problem='zphot_regression', wanted_keys=['HSC', 'DES', 'VHS', 'WISE'], exceptions_keys=['isWISE', 'isVHS','DES_ra', 'DES_dec', 'VHS_class'], cv_ratios=[[80, 20, 20]], constraints='no', zphot_conf_threshold=0.0, zphot_risk_threshold=1.0, zphot_safe_threshold=0.0, zphot_estimator='ephor'):

        self.regression_problem = regression_problem
        self.fits_filename = fits_filename
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
        self.exceptions_keys.append(self.zphot_key)

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

        return

    def process_dataset(self):
        # Select objects of interest from data
        self.reduced_object_zphot, self.nbrobjects_zphot = self.object_selector()

        # Reorder keys
        self.reorder_dict(self.reduced_object_zphot)

        # Compute colors if wanted
        if 'colors' in self.constraints:
            self.reduced_object_zphot = self.color_dict(self.reduced_object_zphot)

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
            tmp_list += list(self.reduced_object_zphot[self.reduced_keys[i]])
            dataset.append(tmp_list)

        self.table_dataset = np.transpose(np.array(dataset, dtype='float'))
        del dataset

        return

    def generate_dataset(self):

        # Save the dataset in a .fits file

        self.nbr_objects = self.table_dataset.shape[0]

        fits_column = []
        for i in range(self.nbr_features):
            if i == self.DES_feature_idx:
                fits_column.append(fits.Column(name=self.reduced_keys[i], array=self.table_dataset[:,i], format='J'))
            else:
                fits_column.append(fits.Column(name=self.reduced_keys[i], array=self.table_dataset[:,i], format='E'))
        dataset = fits.BinTableHDU.from_columns(fits_column)

        del fits_column

        try:
            dataset.writeto(os.path.join(self.savepath, self.regression_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '.fits'))
            print(self.regression_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '.fits generated')
        except OSError as e:
            print(e)

        del dataset

        # Generate cross validation dataset from the full dataset using the generate_cross_testing_datasets function
        self.generate_test_train_val_datasets()

        return

    def compute_constraints_mask(self, constraints):

        constraints_mask = np.zeros(self.data['DES_g'].shape[0], dtype=bool)
        constraints_list =  constraints.split('+')

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
                               (self.data['VHS_j']<19.5) |
                               (self.data['DES_g']<23.2)
                           )
                # isnotbright = [not i for i in isbright]
                constraints_mask += ~isbright
            else:
                print('No constraints defined for "' + i + '" tag')
                print('Continue without applying "' + i + '" constraint')
                continue

        return constraints_mask

    def reorder_dict(self, object_dict):
        if 'noerr' in self.constraints:
            keys_order = ['DES_id', self.zphotconf_key, 'DES_g', 'DES_r', 'DES_i', 'DES_z', 'DES_y']
            if 'VHS' in self.wanted_keys:
                keys_order += ['VHS_j', 'VHS_k']
            if 'WISE' in self.wanted_keys:
                keys_order += ['WISE_w1']
            keys_order += ['z']
        else:
            keys_order = ['DES_id', self.zphotconf_key, 'DES_g', 'DES_r', 'DES_i', 'DES_z', 'DES_y']
            if 'VHS' in self.wanted_keys:
                keys_order += ['VHS_j', 'VHS_k']
            if 'WISE' in self.wanted_keys:
                keys_order += ['WISE_w1']
            keys_order += ['DES_gerr', 'DES_rerr', 'DES_ierr', 'DES_zerr', 'DES_yerr']
            if 'VHS' in self.wanted_keys:
                keys_order += ['VHS_jerr', 'VHS_kerr']
            if 'WISE' in self.wanted_keys:
                keys_order += ['WISE_w1err']
            keys_order += ['z']

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
        object_dict_color['z'] = object_dict['z']

        self.reduced_keys = list(object_dict_color.keys())
        self.nbr_features = len(self.reduced_keys)

        return object_dict_color

    def generate_test_train_val_datasets(self):

        table_dataset_sorted  = self.table_dataset[np.argsort(self.table_dataset[:,-1]), :]

        for i in self.cv_ratios:
            train_ratio = i[0]/100
            test_ratio = i[1]/100
            step_test = int(100/i[1])

            idx_test = range(0,self.nbr_objects,step_test)
            mask_train = np.ones(self.nbr_objects,dtype=bool)
            mask_train[idx_test] = False
            idx_train = np.where(mask_train)[0]
            table_dataset_test = table_dataset_sorted[idx_test, :]
            table_dataset_train = table_dataset_sorted[idx_train,:]
            self.generate_cross_validation_datasets(table_dataset_train)
            fits_column_test = []
            for g in range(self.nbr_features):
                if g == self.DES_feature_idx:
                    fits_column_test.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_test[:,g], format='J'))
                else:
                    fits_column_test.append(fits.Column(name=self.reduced_keys[g], array=table_dataset_test[:,g], format='E'))
            test = fits.BinTableHDU.from_columns(fits_column_test)
            try:
                test.writeto(os.path.join(self.savepath, self.regression_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_test_' + str(i[0]) + '_' + str(i[1]) + '.fits'))
                print(self.regression_problem + '_test_' + str(i[0]) + '_' + str(i[1]) + '.fits generated')
            except OSError as e:
                print(e)

        return

    def generate_cross_validation_datasets(self, table_dataset):

        unique, counts = np.unique(table_dataset[:, -1], return_counts=True)
        class_count_dict_train = dict(zip(unique, counts))
        labels_count_train = list(class_count_dict_train.values())
        nbr_objects_train = table_dataset.shape[0]
        table_dataset_sorted  = table_dataset[np.argsort(table_dataset[:,-1]), :]

        for i in self.cv_ratios:
            train_ratio = i[0]/100
            val_ratio = (i[2]/100)/train_ratio
            nbr_fold = int(100/i[2])
            indexes_chosen_val = np.zeros(nbr_objects_train,dtype=bool)
            for j in range(nbr_fold):
                mask_train = np.ones(nbr_objects_train,dtype=bool)
                table_dataset_train = []
                idx_val = []
                nbr_object_val = []
                avail = np.where(~indexes_chosen_val)[0]
                idx_val += avail[0:avail.shape[0]:nbr_fold].tolist()
                indexes_chosen_val[avail[0:avail.shape[0]:nbr_fold]] = True
                mask_train[avail[0:avail.shape[0]:nbr_fold]] = False
                idx_train = np.where(mask_train)[0]
                table_dataset_val = table_dataset_sorted[idx_val, :]
                table_dataset_train= table_dataset_sorted[idx_train,:]
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
                    val.writeto(os.path.join(self.savepath, self.regression_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_val_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits'))
                    print(self.regression_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_val_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits generated')
                except OSError as e:
                    print(e)
                try:
                    train.writeto(os.path.join(self.savepath, self.regression_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_train_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits'))
                    print(self.regression_problem + '_' + str(self.zphot_safe_threshold) + '-zsafe' + '_train_' + str(i[0]) + '_' + str(i[1]) + '_' + str(j+1) + '.fits generated')
                except OSError as e:
                    print(e)

        return

    def object_selector(self):

        constraints_mask = self.compute_constraints_mask(self.constraints)
        iszphot =   (
                        (self.data['VVDS_zspec'] > 0) |
                        (self.data['DR14QSO_zspec'] > 0) |
                        (self.data[self.zphot_key] >= 0)
                    )
        iszphot = iszphot & ~constraints_mask

        zphot_dict = {n:self.data[n][iszphot] for n in self.data_keys}

        zphot_dict['z'] = []
        for i in range(len(zphot_dict['DES_id'])):
            if (zphot_dict['VVDS_zspec'][i] > 0.0):
                zphot_dict['z'].append(zphot_dict['VVDS_zspec'][i])
                zphot_dict[self.zphotconf_key][i] = 1.0
            elif (zphot_dict['DR14QSO_zspec'][i] > 0.0):
                zphot_dict['z'].append(zphot_dict['DR14QSO_zspec'][i])
                zphot_dict[self.zphotconf_key][i] = 1.0
            else:
                zphot_dict['z'].append(zphot_dict[self.zphot_key][i])
                zphot_dict[self.zphotconf_key][i] = zphot_dict[self.zphotconf_key][i]*(1-zphot_dict[self.zphotrisk_key][i])

        for i in list(zphot_dict.keys()):
            if (i not in self.reduced_keys) and (i != 'z'):
                del zphot_dict[i]

        nbr_zphot = len(self.data[iszphot])
        return zphot_dict, nbr_zphot

    def compute_confidence_value(self, zphot_dict):

        interpolator_conf = interp1d([np.amin(zphot_dict[self.zphotconf_key]),1.0],[0.0,1.0])

        for idx, i in enumerate(zphot_dict[self.zphotconf_key]):
            zphot_dict[self.zphotconf_key][idx] = interpolator_conf(i)

        interpolator_risk = interp1d([np.amin(zphot_dict[self.zphotrisk_key]),1.0],[0.0,1.0])

        for idx, i in enumerate(zphot_dict[self.zphotrisk_key]):
            zphot_dict[self.zphotrisk_key][idx] = interpolator_risk(i)

        for idx, i in enumerate(zphot_dict[self.zphotconf_key]):
            zphot_dict[self.zphotconf_key][idx] = (zphot_dict[self.zphotconf_key][idx] + zphot_dict[self.zphotrisk_key][idx])/2.0

        return zphot_dict

    def Object_selector_zphotnotzero(self):

        ####role####
        # Select QSOs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

        ####inputs####
        # data : Should be the data object outputed by the read_fits function which contains all the data
        # from the .fits file.
        # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

        ####outputs####
        # QSO_dict : A python dictionnary containing all the QSO's data ordered within the same keys as in the data object taken as input.
        # nbr_qso : The number of QSO that were selected
        iszphot =   (
                        (self.data['VVDS_zspec'] > 0) |
                        (self.data['DR14QSO_zspec'] > 0) |
                        (self.data[self.zphot_key]>0)
                    )
        iszphot = iszphot & isbright & ~isoutlier

        iszphot = iszphot & isbright
        zphot_dict = {n:self.data[n][iszphot] for n in self.data_keys}

        zphot_dict['z'] = []
        for i in range(len(zphot_dict['DES_id'])):
            if (zphot_dict['VVDS_zspec'][i] >= 0.0):
                zphot_dict['z'].append(zphot_dict['VVDS_zspec'][i])
                zphot_dict[self.zphotconf_key][i] = 1.0
            elif (zphot_dict['DR14QSO_zspec'][i] >= 0.0):
                zphot_dict['z'].append(zphot_dict['DR14QSO_zspec'][i])
                zphot_dict[self.zphotconf_key][i] = 1.0
            else:
                zphot_dict['z'].append(zphot_dict[self.zphot_key][i])

        for i in list(zphot_dict.keys()):
            if (i not in self.reduced_keys) and (i != 'z'):
                del zphot_dict[i]

        nbr_zphot = len(self.data[iszphot])
        return zphot_dict, nbr_zphot

    def Object_selector_zphotzero(self):

        ####role####
        # Select QSOs from a data object and store them as a python dictionnary with the same keys as the ones of the data object taken as input

        ####inputs####
        # data : Should be the data object outputed by the read_fits function which contains all the data
        # from the .fits file.
        # keys : The keys corresponding to the data object (eg DES_r, HSC_zphot, ...)

        ####outputs####
        # QSO_dict : A python dictionnary containing all the QSO's data ordered within the same keys as in the data object taken as input.
        # nbr_qso : The number of QSO that were selected

        iszphot =   (
                        (self.data['VVDS_zspec'] == 0) |
                        (self.data['DR14QSO_zspec'] == 0) |
                        (self.data[self.zphot_key]==0)
                    )
        iszphot = iszphot & isbright & ~isoutlier
        zphot_dict = {n:self.data[n][iszphot] for n in self.data_keys}

        zphot_dict['z'] = []
        for i in range(len(zphot_dict['DES_id'])):
            if (zphot_dict['VVDS_zspec'][i] == 0.0):
                zphot_dict['z'].append(zphot_dict['VVDS_zspec'][i])
                zphot_dict[self.zphotconf_key][i] = 1.0
            elif (zphot_dict['DR14QSO_zspec'][i] == 0.0):
                zphot_dict['z'].append(zphot_dict['DR14QSO_zspec'][i])
                zphot_dict[self.zphotconf_key][i] = 1.0
            else:
                zphot_dict['z'].append(zphot_dict[self.zphot_key][i])

        for i in list(zphot_dict.keys()):
            if (i not in self.reduced_keys) and (i != 'z'):
                del zphot_dict[i]

        nbr_zphot = len(self.data[iszphot])
        return zphot_dict, nbr_zphot

if __name__ == '__main__':

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # regression_dataset_gen = Regression_dataset_generator(wanted_keys=['HSC', 'DES'], regression_problem='zphot_regression_ephor', exceptions_keys=['isWISE', 'isVHS','DES_ra', 'DES_dec', 'DES_spread', 'VHS_class'], constraints='noout+bright', zphot_estimator='ephor')
    # regression_dataset_gen.process_dataset()
    # regression_dataset_gen.generate_dataset()

    # regression_dataset_gen = Regression_dataset_generator(wanted_keys=['HSC', 'DES', 'VHS', 'WISE'], regression_problem='zphot_regression_ephor', exceptions_keys=['isWISE', 'isVHS','DES_ra', 'DES_dec', 'DES_spread', 'VHS_class'], constraints='noout+bright+nostar', zphot_estimator='ephor')
    # regression_dataset_gen.process_dataset()
    # regression_dataset_gen.generate_dataset()
    #
    # regression_dataset_gen = Regression_dataset_generator(wanted_keys=['HSC', 'DES', 'VHS', 'WISE'], regression_problem='zphot_regression_ephor', exceptions_keys=['isWISE', 'isVHS','DES_ra', 'DES_dec', 'DES_spread', 'VHS_class'], constraints='noout+bright+nostar+zsafe', zphot_safe_threshold=0.3, zphot_estimator='ephor')
    # regression_dataset_gen.process_dataset()
    # regression_dataset_gen.generate_dataset()

    regression_dataset_gen = Regression_dataset_generator(wanted_keys=['HSC', 'DES'], regression_problem='zphot_regression_ephor_noVHS_noWISE', exceptions_keys=['isWISE', 'isVHS','DES_ra', 'DES_dec', 'DES_spread', 'VHS_class'], constraints='noout+bright+nostar+zsafe', zphot_safe_threshold=0.3, zphot_estimator='ephor')
    regression_dataset_gen.process_dataset()
    regression_dataset_gen.generate_dataset()
