import os
import sys
import math
import warnings
import numpy as np
from astropy.io import fits
from dataset_generator_regression import Regression_dataset_generator
from dataset_generator_classification import Classification_dataset_generator

from utils_regression import *
from utils_classification import *

def classification_regression_crossover(classification_table_dataset, regression_table_dataset, crossover_ratio):

    # crossover_ratio (float) : Fraction of objects shred among the two dataset to keep in classification

    unique, counts = np.unique(classification_table_dataset[:, -1], return_counts=True)
    class_count_dict_classification = dict(zip(unique, counts))
    print('Before crossover : ', class_count_dict_classification)
    nbr_others_classification = class_count_dict_classification[0.0]
    nbr_others_regression = 0
    regression_crossover_mask = np.ones((regression_table_dataset.shape[0]), dtype=bool)
    classification_crossover_mask = np.ones((classification_table_dataset.shape[0]), dtype=bool)

    # Get all indexes from classification where label is not others
    indexes_classification_no_others = np.where(classification_table_dataset[:,-1] != 0.0)[0]
    # Get index of all object id from classification table shared with regression table with others label (ie class==0.0)
    DES_id_match_indexes_classification = np.where(np.in1d(classification_table_dataset[:,0],regression_table_dataset[:,0])*(classification_table_dataset[:,-1] == 0.0))[0]
    # Get indexes of all object id from classification table not shared with regression table with others label (ie class==0.0)
    DES_id_nomatch_indexes_classification = np.where(~(np.in1d(classification_table_dataset[:,0],regression_table_dataset[:,0]))*(classification_table_dataset[:,-1] == 0.0))[0]
    # Get indexes of all object id from regression table shared with classification table with others label (ie class==0.0)
    DES_id_match_indexes_regression = np.where(np.in1d(regression_table_dataset[:,0],classification_table_dataset[DES_id_match_indexes_classification,0]))[0]

    if crossover_ratio > 0.0:
        fraction_idx = math.floor(len(DES_id_match_indexes_regression)*crossover_ratio)
        index_selection_classification = list(DES_id_match_indexes_classification[0:fraction_idx]) + list(indexes_classification_no_others) + list(DES_id_nomatch_indexes_classification)
        index_selection_regression = list(DES_id_match_indexes_regression[fraction_idx:])
    else:
        index_selection_classification = list(indexes_classification_no_others) + list(DES_id_nomatch_indexes_classification)
        index_selection_regression = list(DES_id_match_indexes_regression)

    classification_table_dataset = classification_table_dataset[index_selection_classification, :]
    regression_table_dataset = regression_table_dataset[index_selection_regression, :]

    unique, counts = np.unique(classification_table_dataset[:, -1], return_counts=True)
    class_count_dict_classification = dict(zip(unique, counts))
    print('After crossover : ', class_count_dict_classification)

    return classification_table_dataset, regression_table_dataset

if __name__ == '__main__':

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    classification_dataset_gen = Classification_dataset_generator(wanted_keys=['DES', 'VHS', 'WISE', 'HSC'], classification_problem='BG_LRG_ELG_QSO_classification_ephor_crossover', exceptions_keys=['isWISE', 'isVHS','DES_ra', 'DES_dec', 'VHS_class'], constraints='noout+bright+nostar+zsafe', zphot_safe_threshold=0.3, zphot_estimator='ephor')
    classification_dataset_gen.process_dataset()

    regression_dataset_gen = Regression_dataset_generator(wanted_keys=['DES', 'VHS', 'WISE', 'HSC'], regression_problem='zphot_regression_ephor_crossover', exceptions_keys=['isWISE', 'isVHS','DES_ra', 'DES_dec', 'DES_spread', 'VHS_class'], constraints='noout+bright+nostar+zsafe', zphot_safe_threshold=0.3, zphot_estimator='ephor')
    regression_dataset_gen.process_dataset()

    classification_dataset_gen.table_dataset, regression_dataset_gen.table_dataset = classification_regression_crossover(classification_dataset_gen.table_dataset, regression_dataset_gen.table_dataset, 0.5)
    classification_dataset_gen.generate_dataset()
    regression_dataset_gen.generate_dataset()
