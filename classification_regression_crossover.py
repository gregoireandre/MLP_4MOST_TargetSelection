import __init__
import preprocessing_classification
import utils_classification
import utils_regression
from regression import Regression

import os
import sys
import math
import json
import pprint
import random
import datetime
import subprocess
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits


class Classification():

    def __init__(self, classifier='ANN', preprocessing_method=None, random_seed=7, catalog_filename='4MOST.CatForGregoire.11Oct2018.zphot', classification_problem='BG_LRG_ELG_QSO_classification', constraints='no', data_imputation=0.0, normalization='cr', train_test_val_split=[80, 20, 20], cv_fold_nbr=1, cross_validation=False, tsboard=False, early_stop=True, lrdecay=True, compute_conf_matr=False, model_path=None, model_index=None, ANN_zphot=None):

        self.classifier = classifier                                            # Classifier (str) : 'ANN', 'RF', 'SVM'
        self.preprocessing_method = preprocessing_method                        # Processing method (dict) : {'method' : 'method to use among preprocessing.py', 'arguments': [arg1, arg2, arg3, ...]}
        self.random_seed = random_seed                                          # Random seed (int) : Random seed to use for reproducibility
        self.catalog_filename = catalog_filename                                # Catalog Filename (str) : Filename of the catalog to use (must be a .fits and located in "src" folder)
        self.classification_problem = classification_problem                    # Classification problem (str) : Classification problem to consider (see dataset_generator.py for details about different classification problem)
        self.constraints= constraints                                           # Constraints (str or dict) : Constraints to use on the dataset (see dataset_generator.py for details about different classification problem)
        self.data_imputation = data_imputation                                  # Data imputation (float) : Value used while generating the dataset to fill empty entries
        self.normalization = normalization
        self.train_test_val_split = train_test_val_split                    # Train Test split (list) : Fraction of dataset to use on training/testing and on training/validation (e.g [80, 20])
        self.cv_fold_nbr = cv_fold_nbr                                          # Cross Validation fold number (int) : The index of the cross validation dataset to use (refer to dataset_generator.py for more info)
        self.cross_validation = cross_validation                                # Cross Validation (bool) : Weither or not use cross validation during evaluation of the model
        self.tsboard = tsboard                                                  # Tensorboard (bool) : Weither or not use tensorboard callback during training of the model
        self.early_stop = early_stop                                            # Early Stop (bool) : Weither or not use early stop callback during training of the model
        self.lrdecay = lrdecay                                                  # Learning Rate decay (bool) : Weither or not use learning rate plateau decay callback during training of the model
        self.compute_conf_matr = compute_conf_matr                              # Compute Confusion Matrix (bool) : Weither or not compute confusion matrix and save to png during model evalutation
        self.model_index = model_index
        self.model_path = model_path
        self.ANN_zphot = ANN_zphot

        np.random.seed(self.random_seed)                                        # Fix Numpy random seed for reproducibility
        self.script_path = os.path.realpath(__file__)
        self.script_dir, _ = os.path.split(self.script_path)
        self.constraints = utils_classification.constraints_to_str(self.constraints)                 # Convert constraints input varibale to str
        self.classnames = utils_classification.compute_classnames(self.classification_problem)

        self.input_dict = {
                            'classifier': classifier,
                            'preprocessing_method': preprocessing_method,
                            'random_seed': random_seed,
                            'catalog_filename': catalog_filename,
                            'classification_problem': classification_problem,
                            'constraints': constraints,
                            'data_imputation': data_imputation,
                            'normalization': normalization,
                            'train_test_val_split': train_test_val_split,
                            'cv_fold_nbr': cv_fold_nbr,
                            'cross_validation': cross_validation,
                            'tsboard': tsboard,
                            'early_stop': early_stop,
                            'lrdecay': lrdecay,
                            'compute_conf_matr': compute_conf_matr,
                            'random_seed': random_seed,
                            'constraints': constraints
                          }

        self.dataset_path = os.path.join(self.script_dir, 'datasets', self.catalog_filename, self.constraints + '-constraints')

        # Create model_ANN folder in "src" directory

        if not os.path.exists(os.path.join(self.script_dir, 'model_' + self.classifier + '_classification')):
            os.makedirs(os.path.join(self.script_dir, 'model_' + self.classifier + '_classification'))

        # To allow effective comparison of increasing number of model, the models are named iteratively with an integer beginning at 1.
        # A folder named after the model is then created and all the information realtive to the model and its performance is stored in the latter.

        if model_index is None:
            # Compute model filename by counting the number of folder in the "model_ANN" directory
            nbr_directories = len(next(os.walk(os.path.join(self.script_dir, 'model_' + self.classifier + '_classification')))[1])
            self.model_index = nbr_directories+1

        if model_path is None:
            self.model_path = os.path.join(self.script_dir, 'model_' + self.classifier + '_classification', str(self.model_index))
            os.makedirs(self.model_path)

        print('Initialized, will load dataset')
        print('Class Labels :')
        for idx, i in enumerate(self.classnames):
            print(i + ': ' + str(idx))

    def run(self):

        if self.classifier == 'ANN':
            self.define_ANN()
            self.init_ANN()
            self.train_ANN()
            self.evaluate_ANN()

        return

    def load_dataset(self):

        self.DES_id_train, self.X_train, self.Y_train, self.sample_weights_train, self.DES_id_val, self.X_val, self.Y_val, self.sample_weights_val, self.DES_id_test, self.X_test, self.Y_test, self.sample_weights_test, self.data_keys = utils_classification.load_dataset(self.dataset_path, self.model_path, self.classification_problem, self.train_test_val_split, self.cv_fold_nbr, self.normalization, self.data_imputation, self.random_seed)
        # self.X_train_norm, self.X_val_norm, self.X_test_norm, self.data_imputation_values = utils_classification.fill_empty_entries_dataset(self.X_train, self.X_val,self. X_test, self.data_imputation, self.constraints)
        # self.X_train_norm, self.X_val_norm, self.X_test_norm, self.normalizer = utils_classification.normalize_dataset(self.X_train_norm, self.X_val_norm, self.X_test_norm, self.normalization, self.random_seed)
        self.X_train_norm, self.X_val_norm, self.X_test_norm, self.normalizer = utils_classification.normalize_dataset(self.X_train, self.X_val,self. X_test, self.normalization, self.random_seed)
        self.X_train_norm, self.X_val_norm, self.X_test_norm, self.data_imputation_values = utils_classification.fill_empty_entries_dataset(self.X_train_norm, self.X_val_norm, self.X_test_norm, self.data_imputation, self.constraints)
        if self.preprocessing_method is not None:
            already_applied, to_apply = self.check_existing_preprocessed_datasets()
            if already_applied:
                print('Loading following preprocessed dataset : ', already_applied)
                self.load_preprocessed_dataset(already_applied)
            if to_apply:
                print('Processing dataset with : ', to_apply)
                for i in to_apply:
                    if len(i['arguments']) == 1:
                        self.X_train_norm, self.Y_train = getattr(preprocessing_classification, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0])
                    elif len(i['arguments']) == 2:
                        self.X_train_norm, self.Y_train = getattr(preprocessing_classification, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1])
                    elif len(i['arguments']) == 3:
                        self.X_train_norm, self.Y_train = getattr(preprocessing_classification, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1], i['arguments'][2])
                    elif len(i['arguments']) == 4:
                        self.X_train_norm, self.Y_train = getattr(preprocessing_classification, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1], i['arguments'][2], i['arguments'][3])
                preprocessing_classification.save_preprocessed_dataset(self.script_dir, self.catalog_filename, self.constraints, self.data_imputation, self.normalization, self.classification_problem, self.train_test_val_split, self.cv_fold_nbr, self.preprocessing_method, self.X_train_norm, self.Y_train)

        # if self.ANN_zphot is not None:
        #     self.X_train_norm = utils_classification.unapply_reduce_center(self.X_train_norm, self.normalizer)
        #     self.X_val_norm = utils_classification.unapply_reduce_center(self.X_val_norm, self.normalizer)
        #     self.compute_zphot_regressor('ANN', 126)
        #
        # self.X_train_norm[:, :-2] = utils_classification.apply_normalization(self.X_train_norm[:, :-2], self.normalization, self.normalizer)
        # self.X_train_norm[:, -2] = (self.X_train_norm[:, -2] - np.mean(self.X_train_norm[:, -2]))/np.std(self.X_train_norm[:, -2])
        # self.X_train_norm[:, -1] = (self.X_train_norm[:, -1] - np.mean(self.X_train_norm[:, -1]))/np.std(self.X_train_norm[:, -1])
        #
        # self.X_val_norm[:, :-2] = utils_classification.apply_normalization(self.X_val_norm[:, :-2], self.normalization, self.normalizer)
        # self.X_val_norm[:, -2] = (self.X_val_norm[:, -2] - np.mean(self.X_val_norm[:, -2]))/np.std(self.X_val_norm[:, -2])
        # self.X_val_norm[:, -1] = (self.X_val_norm[:, -1] - np.mean(self.X_val_norm[:, -1]))/np.std(self.X_val_norm[:, -1])

        # preprocessing_classification.save_preprocessed_dataset(self.script_dir, self.catalog_filename, self.constraints + '+ANNzphot', self.data_imputation, self.normalization, self.classification_problem, self.train_test_val_split, self.cv_fold_nbr, self.preprocessing_method, self.X_train_norm, self.Y_train)

        self.input_dimensions = self.X_train_norm.shape[1]
        self.nbr_classes = np.unique(self.Y_test).shape[0]
        return

    def check_existing_preprocessed_datasets(self):

        preprocessed_dataset_path = os.path.join(self.dataset_path, 'preprocessed_' + str(self.data_imputation) + '-imputation', self.classification_problem + '_train_' + str(self.train_test_val_split[0]) + '_' + str(self.train_test_val_split[1]) + '_' + str(self.cv_fold_nbr) + '_norm-' + str(self.normalization))
        preprocessing_methods = self.preprocessing_method.copy()
        to_apply = []
        already_applied = []
        flag = True

        while flag:

            for idx, i in enumerate(preprocessing_methods):
                if idx == 0:
                    preprocessed_dataset_filename = i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])
                else:
                    preprocessed_dataset_filename += '_' + i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])

            full_filename = os.path.join(preprocessed_dataset_path, preprocessed_dataset_filename + '.fits')

            if not os.path.isfile(full_filename):
                to_apply = [preprocessing_methods.pop()] + to_apply
                if not preprocessing_methods:
                    flag=False
            else:
                already_applied = preprocessing_methods
                flag=False

        return already_applied, to_apply

    def load_preprocessed_dataset(self, preprocessing_methods):

        preprocessed_dataset_path = os.path.join(self.dataset_path, 'preprocessed_' + str(self.data_imputation) + '-imputation', self.classification_problem + '_train_' + str(self.train_test_val_split[0]) + '_' + str(self.train_test_val_split[1]) + '_' + str(self.cv_fold_nbr) + '_norm-' + str(self.normalization))
        for idx, i in enumerate(preprocessing_methods):
            if idx == 0:
                preprocessed_dataset_filename = i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])
            else:
                preprocessed_dataset_filename += '_' + i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])

        self.training_dataset, _ = utils_classification.read_fits(os.path.join(preprocessed_dataset_path, preprocessed_dataset_filename + '.fits'))
        np.random.shuffle(self.training_dataset)

        # split into input (X) and output (Y) variables
        self.X_train_os = self.training_dataset[:,:-1]
        self.Y_train = self.training_dataset[:,-1]

        self.input_dimensions = self.X_train_os.shape[1]
        self.nbr_classes = np.unique(self.Y_val).shape[0]

        return

    def define_ANN(self):

        # Load kerass modules and set them to global variable (this allows to load tensorflow only if it is needed due to the fact that as soon as tensorflow is loaded, the GPU is fully allocated to the latter)

        global keras
        import keras
        global set_random_seed
        from tensorflow import set_random_seed
        global classification_report
        from sklearn.metrics import classification_report
        global selu
        from keras.activations import selu
        global Sequential
        from keras.models import Sequential
        global l1, l2
        from keras.regularizers import l1, l2
        global Adam, SGD
        from keras.optimizers import Adam, SGD
        global lecun_normal
        from keras.initializers import lecun_normal
        global Dense, Dropout, AlphaDropout, BatchNormalization
        from keras.layers import Dense, Dropout, AlphaDropout, BatchNormalization
        global ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
        from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

        set_random_seed(self.random_seed)

        # The parameters of the ANN are stored in a dictionnary

        self.ann_parameters =  {'loss_function': 'categorical_crossentropy',
                                'learning_rate': 0.001,
                                'batch_size': 256,
                                'epochs': 400,
                                'metrics': ['categorical_accuracy'],
                                'nbr_layers': 3,
                                'nbr_neurons' : 128,
                                'activation' : 'tanh',
                                'output_activation': 'softmax',
                                'dropout_strength': 0.25,
                                'kernel_initializer': 'lecun_normal',
                                'bias_initializer' : 'zeros',
                                'kernel_regularizer': None,
                                'bias_regularizer': None,
                                'activity_regularizer': None,
                                'kernel_constraint': None,
                                'bias_constraint': None,
                                'SNN': False,
                                'weighted': True,
                                'batch_normalization': False}

        self.ann_parameters['optimizer'] = Adam(lr=self.ann_parameters['learning_rate'])
        self.ann_parameters['optimizer_str'] = 'Adam'
        if self.ann_parameters['kernel_regularizer'] is not None:
            regularizer_dict = self.ann_parameters['kernel_regularizer'].__dict__
            for i in list(regularizer_dict.keys()):
                if regularizer_dict[i] > 0.0:
                    regularizer_str = i
                    regularization_strength = regularizer_dict[i]
            self.ann_parameters['kernel_regularizer_str'] = regularizer_str
            self.ann_parameters['regularization_strength'] = regularization_strength
        else:
            self.ann_parameters['kernel_regularizer_str'] = None
            self.ann_parameters['regularization_strength'] = 0.0
        if self.ann_parameters['dropout_strength'] > 0.0:
            self.ann_parameters['dropout'] = True
        else:
            self.ann_parameters['dropout'] = False


        # The architecture of the ANN is computed from the ANN parameters dictionnary and stored in a list of layers

        self.layers_list = []
        self.layers_list.append({'size': self.ann_parameters['nbr_neurons'], 'input_dim': self.input_dimensions, 'activation': self.ann_parameters['activation']})
        for i in range(1,self.ann_parameters['nbr_layers'] - 1):
            self.layers_list.append({'size': self.ann_parameters['nbr_neurons'], 'activation' : self.ann_parameters['activation']})
        self.layers_list.append({'size': self.nbr_classes, 'activation': self.ann_parameters['output_activation']})

        return

    def run_ANN_gridsearch(self, max_run=250):

        global keras
        import keras
        global set_random_seed
        from tensorflow import set_random_seed
        global classification_report
        from sklearn.metrics import classification_report
        global selu
        from keras.activations import selu
        global Sequential
        from keras.models import Sequential
        global l1, l2
        from keras.regularizers import l1, l2
        global Adam, SGD
        from keras.optimizers import Adam, SGD
        global lecun_normal
        from keras.initializers import lecun_normal
        global Dense, Dropout, AlphaDropout, BatchNormalization
        from keras.layers import Dense, Dropout, AlphaDropout, BatchNormalization
        global ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
        from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

        set_random_seed(self.random_seed)

        # The parameters of the ANN are stored in a dictionnary

        self.ann_parameters_talos =  {'loss_function': ['categorical_crossentropy'],
                                      'learning_rate': [0.001],
                                      'batch_size': [256],
                                      'epochs': [200],
                                      'metrics': [['categorical_accuracy']],
                                      'nbr_layers': [3],
                                      'nbr_neurons' : [128],
                                      'input_activation' : ['tanh'],
                                      'activation' : ['tanh'],
                                      'output_activation': ['softmax'],
                                      'dropout_strength': [0.0],
                                      'kernel_initializer': ['lecun_normal'],
                                      'bias_initializer' : ['zeros'],
                                      'kernel_regularizer': [None],
                                      'bias_regularizer': [None],
                                      'activity_regularizer': [None],
                                      'kernel_constraint': [None],
                                      'bias_constraint': [None],
                                      'weighted': [False],
                                      'batch_normalization': [False]}

        all_keys = list(self.ann_parameters_talos.keys())
        all_combinations = list(itertools.product(*(self.ann_parameters_talos[key] for key in all_keys)))
        print('ANN GridSearch has ' + str(len(all_combinations)) + ' combinations')
        if len(all_combinations) > max_run:
            print('Randomly elect ' + str(max_run) + ' among them')
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_run]
        for idx,combination in enumerate(all_combinations):
            if idx > 0:
                nbr_directories = len(next(os.walk(os.path.join(self.script_dir, 'model_' + self.classifier + '_classification')))[1])
                self.model_index = nbr_directories+1
                self.model_path = os.path.join(self.script_dir, 'model_' + self.classifier + '_classification', str(self.model_index))
                os.makedirs(self.model_path)
            self.load_dataset()
            self.ann_parameters = dict(zip(all_keys, combination))
            if self.ann_parameters['kernel_regularizer'] is not None:
                regularizer_dict = self.ann_parameters['kernel_regularizer'].__dict__
                for i in list(regularizer_dict.keys()):
                    if regularizer_dict[i] > 0.0:
                        regularizer_str = i
                        regularization_strength = float(regularizer_dict[i])
                self.ann_parameters['kernel_regularizer_str'] = regularizer_str
                self.ann_parameters['regularization_strength'] = regularization_strength
            else:
                self.ann_parameters['kernel_regularizer_str'] = None
                self.ann_parameters['regularization_strength'] = 0.0
            if self.ann_parameters['dropout_strength'] > 0.0:
                self.ann_parameters['dropout'] = True
            else:
                self.ann_parameters['dropout'] = False

            self.ann_parameters['optimizer'] = Adam(lr=self.ann_parameters['learning_rate'])
            self.ann_parameters['optimizer_str'] = 'Adam'

            print('Run with following parameters :')
            pprint.pprint(self.ann_parameters, width=1)

            # The architecture of the ANN is computed from the ANN parameters dictionnary and stored in a list of layers

            self.layers_list = []
            self.layers_list.append({'size': self.ann_parameters['nbr_neurons'], 'input_dim': self.input_dimensions, 'activation': self.ann_parameters['input_activation']})
            for i in range(1,self.ann_parameters['nbr_layers'] - 1):
                self.layers_list.append({'size': self.ann_parameters['nbr_neurons'], 'activation' : self.ann_parameters['activation']})
            self.layers_list.append({'size': self.nbr_classes, 'activation': self.ann_parameters['output_activation']})

            self.init_ANN()
            self.train_ANN()
            self.evaluate_ANN()
        return

    def init_ANN(self):

        model = Sequential()
        for idx,i in enumerate(self.layers_list):
            if i['activation'] in ['PReLU', 'LeakyReLU']:
                if i['activation'] == 'PReLU':
                    act = keras.layers.PReLU()
                if i['activation'] == 'LeakyReLU':
                    act = keras.layers.LeakyReLU()
                if 'input_dim' in i.keys():
                    model.add(Dense(i['size'], input_dim=i['input_dim']))
                    model.add(act)
                    if self.ann_parameters['dropout']:
                        model.add(Dropout(self.ann_parameters['dropout_strength'], None, self.random_seed))
                    if self.ann_parameters['batch_normalization']:
                        model.add(BatchNormalization())
                elif(idx == len(self.layers_list) - 1):
                    model.add(Dense(i['size']))
                    model.add(act)
                else:
                    model.add(Dense(i['size']))
                    model.add(act)
                    if self.ann_parameters['dropout']:
                        model.add(Dropout(self.ann_parameters['dropout_strength'], None, self.random_seed))
                    if self.ann_parameters['batch_normalization']:
                        model.add(BatchNormalization())
            else:
                if 'input_dim' in i.keys():
                    model.add(Dense(i['size'], input_dim=i['input_dim'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['dropout']:
                        model.add(Dropout(self.ann_parameters['dropout_strength'], None, self.random_seed))
                    if self.ann_parameters['batch_normalization']:
                        model.add(BatchNormalization())
                elif(idx == len(self.layers_list) - 1):
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                else:
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['dropout']:
                        model.add(Dropout(self.ann_parameters['dropout_strength'], None, self.random_seed))
                    if self.ann_parameters['batch_normalization']:
                        model.add(BatchNormalization())

        self.model = model
        self.model.compile(loss=self.ann_parameters['loss_function'], optimizer=self.ann_parameters['optimizer'], metrics=self.ann_parameters['metrics'] + [f1])

        return

    def train_ANN(self):

        # Import custom metrics from custom_metrics.py

        from custom_metrics import Metrics

        if not os.path.exists(os.path.join(self.model_path, 'tsboard')):
            os.makedirs(os.path.join(self.model_path, 'tsboard'))

        if not os.path.exists(os.path.join(self.model_path, 'performances', 'val')):
            os.makedirs(os.path.join(self.model_path, 'performances', 'val'))

        if not os.path.exists(os.path.join(self.model_path, 'checkpoints')):
            os.makedirs(os.path.join(self.model_path, 'checkpoints'))

        # One hot encoding of labels to match the use of softmax activation function in the last layer and categorical crossentropy loss function
        # One hot encoding means that if our label can take 5 different values (e.g 0,1,2,3,4) then the labels are transformed in vectors in 5 dimensional space
        # Example, a label of 0 is equivalent to [1, 0, 0, 0, 0] in one hot representation
        #          a label of 3 is equivalent to [0, 0, 0, 1, 0] in one hot representation

        self.Y_train, self.Y_val, self.Y_test = utils_classification.one_hot_encode(self.Y_train, self.Y_val, self.Y_test)
        self.custom_metrics = Metrics()

        # Store the ANN parameters in a json file and save them in the model folder

        with open(os.path.join(self.model_path, 'classification_inputs.json'), 'w') as fp:
            json.dump(self.input_dict, fp)

        ann_parameters_json = self.ann_parameters.copy()
        del ann_parameters_json['optimizer']
        del ann_parameters_json['kernel_regularizer']
        ann_parameters_json['metrics'] = ann_parameters_json['metrics'][0]
        with open(os.path.join(self.model_path, 'ANN_parameters.json'), 'w') as fp:
            json.dump(ann_parameters_json, fp)

        # serialize model architecture to JSON
        model_json = self.model.to_json()
        with open(os.path.join(self.model_path, 'ANN_architecture.json'), "w") as fp:
            fp.write(json.dumps(model_json))

        checkpoints_name = ('ANN_checkpoints_'
                           + '{epoch:03d}-epo'
                           + '.hdf5')

        # Define callbacks to use between each epoch during training

        # ModelCheckpoint saves the weights of the neural network at each epoch
        checkpoint = ModelCheckpoint(os.path.join(self.model_path, 'checkpoints', checkpoints_name), verbose=2, save_best_only=False)
        # ReduceLROnPlateau reduces learning rate by a factor when the quantity monitored does not decreases for a given number of epoch (patience)
        lrdecay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.000001, min_delta=0.0001, cooldown=5, verbose=1)
        # lrdecay =  lr_decay_custom_callback(self.ann_parameters['learning_rate'], 0.9625, 0.000000001)
        # EarlyStopping stops training of the ANN if the quantity monitored does not get better for a given number of epoch (patience)
        earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=20, verbose=1, mode='min')
        # TensorBoard callback generates at each epochs the tensorboard files
        tsboard = TensorBoard(log_dir=os.path.join(self.model_path, 'tsboard'), histogram_freq=1, batch_size=self.ann_parameters['batch_size'], write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

        # Compute callbacks list from callbacks
        callbacks_list = [checkpoint, self.custom_metrics]

        if self.tsboard:
            callbacks_list += [tsboard]
        if self.early_stop:
            callbacks_list += [earlystop]
        if self.lrdecay:
            callbacks_list += [lrdecay]

        if self.ann_parameters['weighted']:
            print('Using weights.')
            print('Training weighths: ', self.sample_weights_train, self.sample_weights_train.shape)
            print('Validation weighths: ', self.sample_weights_val, self.sample_weights_val.shape)
            self.history = self.model.fit(self.X_train_norm, self.Y_train, sample_weight=self.sample_weights_train, validation_data=(self.X_val_norm, self.Y_val, self.sample_weights_val), epochs=self.ann_parameters['epochs'], batch_size=self.ann_parameters['batch_size'], callbacks=callbacks_list, verbose=2)
        else:
            self.history = self.model.fit(self.X_train_norm, self.Y_train, validation_data=(self.X_val_norm, self.Y_val), epochs=self.ann_parameters['epochs'], batch_size=self.ann_parameters['batch_size'], callbacks=callbacks_list, verbose=2)

        # Loop over all checkpoints and keep only the one with best metric. Then rename it with its metric score in the filename

        last_f1_score = self.custom_metrics.macro_f1s[-1]
        best_f1_score = max(self.custom_metrics.macro_f1s)
        best_f1_epoch = self.custom_metrics.epochs[self.custom_metrics.macro_f1s.index(best_f1_score)]

        directories = os.listdir(os.path.join(self.model_path, 'checkpoints'))
        for idx, i in enumerate(directories):
            if '{:03}'.format(best_f1_epoch) in i:
                new_filename = i.split('.')[0] + '_' + str(best_f1_score) + '-macrof1.hdf5'
                os.rename(os.path.join(self.model_path, 'checkpoints', i), os.path.join(self.model_path, 'checkpoints', new_filename))
            else:
                os.remove(os.path.join(self.model_path, 'checkpoints', i))

        # Keep in attributes weither the training was stopped due to earlystop callback or not.
        # If earlystop is called suring training earlystop.stopped_epoch is the epoch at which the training was stopped
        # If earlystop is not called during taining earlystop.stopped_epoch = 0

        self.early_stopped_epoch = earlystop.stopped_epoch

        # Save weights and store last epoch as well as metric score in the filename of the weights file

        if self.early_stopped_epoch > 0:
            # serialize weights to HDF5
            self.model.save_weights(os.path.join(self.model_path, 'ANN_weights_' + str(self.early_stopped_epoch) + '-epo_' + str(last_f1_score) + '-macrof1.hdf5'))
        else:
            self.model.save_weights(os.path.join(self.model_path, 'ANN_weights_' + str(self.ann_parameters['epochs']) + '-epo_' + str(last_f1_score) + '-macrof1.hdf5'))

        print("Saved model to disk")

        # Plot model loss function and accuracy across epochs
        # accuracy
        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.macro_f1s)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.macro_precision)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.macro_recall)
        plt.title('Model Performance')
        plt.ylabel('Validation Score')
        plt.xlabel('epoch')
        plt.legend(['macro f1', 'macro precision', 'macro recall'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'validation_score.png'))

        plt.clf()

        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.no_others_f1)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.no_others_precision)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.no_others_recall)
        plt.title('Model Performance no others')
        plt.ylabel('Validation Score')
        plt.xlabel('epoch')
        plt.legend(['macro f1', 'macro precision', 'macro recall'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'validation_score_no_others.png'))

        plt.clf()

        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.elgs['f1'])
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.elgs['precision'])
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.elgs['recall'])
        plt.plot(self.custom_metrics.epochs, [0.85 for i in range(len(self.custom_metrics.epochs))])
        plt.title('Model Performance on ELGs')
        plt.ylabel('score')
        plt.xlabel('epoch')
        plt.legend(['f1-score', 'precision', 'recall', 'f1-score-colorcut'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'ELGs_score.png'))

        plt.clf()

        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.lrgs['f1'])
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.lrgs['precision'])
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.lrgs['recall'])
        plt.plot(self.custom_metrics.epochs, [0.895 for i in range(len(self.custom_metrics.epochs))])
        plt.title('Model Performance on LRGs')
        plt.ylabel('score')
        plt.xlabel('epoch')
        plt.legend(['f1-score', 'precision', 'recall', 'f1-score-colorcut'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'LRGs_score.png'))

        plt.clf()

        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.bgs['f1'])
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.bgs['precision'])
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.bgs['recall'])
        plt.plot(self.custom_metrics.epochs, [0.995 for i in range(len(self.custom_metrics.epochs))])
        plt.title('Model Performance on BGs')
        plt.ylabel('Validation Score')
        plt.xlabel('epoch')
        plt.legend(['f1-score', 'precision', 'recall', 'f1-score-colorcut'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'BGs_score.png'))

        plt.clf()

        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.qsos['f1'])
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.qsos['precision'])
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.qsos['recall'])
        plt.title('Model Performance on QSOs')
        plt.ylabel('Validation Score')
        plt.xlabel('epoch')
        plt.legend(['f1-score', 'precision', 'recall', 'f1-score-colorcut'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'QSOs_score.png'))

        plt.clf()

        # loss
        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'ANN_loss.png'))

        plt.close()

        return

    def evaluate_ANN(self):

        # evaluate the model
        Y_pred = self.model.predict(self.X_val_norm)

        savepath = os.path.join(self.model_path, 'performances', 'val')

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if self.compute_conf_matr:
            mean_auc_roc, mean_auc_pr = utils_classification.compute_aucs(Y_pred, self.Y_val, self.DES_id_val, self.classnames, savepath, plot=True)
        else:
            mean_auc_roc, mean_auc_pr = utils_classification.compute_aucs(Y_pred, self.Y_val, self.DES_id_val, self.classnames)

        if Y_pred.shape[1] > 1:
            Y_pred = np.argmax(Y_pred, axis=-1)
            Y_val = np.argmax(self.Y_val, axis=-1)
        else:
            Y_pred = np.squeeze(Y_pred)
            Y_pred = (Y_pred > 0.5).astype(int)

        report = classification_report(Y_val, Y_pred, target_names=self.classnames)
        filename_report = 'ANN_classification_validation_report.txt'
        with open(os.path.join(self.model_path, filename_report), "w") as fp:
            fp.write(report)

        utils_classification.report2csv(self.model_index, os.path.join(self.script_dir, 'model_' + self.classifier + '_classification'), self.catalog_filename, self.constraints, self.ann_parameters, self.classification_problem, self.train_test_val_split, self.cv_fold_nbr, self.data_imputation, self.normalization, self.model_path, self.preprocessing_method, self.early_stopped_epoch, mean_auc_roc, mean_auc_pr, self.custom_metrics)

        return

    def load_ANN(self, weights_flag):

        global keras
        import keras
        global Adam, SGD
        from keras.optimizers import Adam, SGD
        global model_from_json
        from keras.models import model_from_json
        global set_random_seed
        from tensorflow import set_random_seed

        set_random_seed(self.random_seed)

        with open(os.path.join(self.model_path, 'ANN_architecture.json')) as f:
            ANN_architecture = json.load(f)
        with open(os.path.join(self.model_path,'ANN_parameters.json')) as f:
            ANN_parameters = json.load(f)

        self.model = model_from_json(ANN_architecture)
        self.model.load_weights(utils_classification.get_model_weights_path(self.model_path, weights_flag))
        self.model.compile(loss=ANN_parameters['loss_function'], optimizer=Adam(lr=ANN_parameters['learning_rate']), metrics=[])

        return

    def init_RF(self):

        rf = RandomForestClassifier(random_state=self.random_seed, verbose=1)
        self.model = rf
        return

    def init_SVM(self):

        svc = SVC(random_state=self.random_seed, verbose=1)
        self.model = svc
        return

    def CV_gridsearch(self, params_dist, metric):

        sys.stdout = open(os.path.join(self.script_dir, 'model_' + self.classifier, 'CV_gridsearch_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +'.txt' , 'w'))

        print("# Tuning hyper-parameters for %s" % metric)
        print()

        clf = GridSearchCV(self.model, params_dist, n_jobs=1, cv=10,
                           scoring='%s_macro' % metric)
        clf.fit(self.X_train_norm, self.Y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = self.Y_val, clf.predict(self.X_val_norm)
        print(classification_report(y_true, y_pred))
        print()

        sys.stdout.close()

        return

    def compute_zphot_regressor(self, regressor, model_index_zphot, weights_flag='final'):

        script_path = os.path.realpath(__file__)
        script_dir, _ = os.path.split(script_path)
        model_path = os.path.join(script_dir, 'model_' + regressor + '_regression', str(model_index_zphot))

        with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
            ANN_architecture = json.load(f)
        with open(os.path.join(model_path,'ANN_parameters.json')) as f:
            ANN_parameters = json.load(f)
        with open(os.path.join(model_path,'regression_inputs.json')) as f:
            regression_inputs = json.load(f)

        restored_object_zphot = Regression(regressor='ANN',
                                         preprocessing_method=regression_inputs['preprocessing_method'],
                                         catalog_filename=regression_inputs['catalog_filename'],
                                         regression_problem=regression_inputs['regression_problem'],
                                         constraints=regression_inputs['constraints'],
                                         data_imputation=regression_inputs['data_imputation'],
                                         normalization=regression_inputs['normalization'],
                                         train_test_val_split=regression_inputs['train_test_val_split'],
                                         cv_fold_nbr=regression_inputs['cv_fold_nbr'],
                                         cross_validation=regression_inputs['cross_validation'],
                                         tsboard=regression_inputs['tsboard'],
                                         early_stop=regression_inputs['early_stop'],
                                         lrdecay=regression_inputs['lrdecay'],
                                         zphot_conf_threshold=regression_inputs['zphot_conf_threshold']
                                        )
        restored_object_zphot.load_dataset()
        del restored_object_zphot.X_val
        del restored_object_zphot.X_train
        del restored_object_zphot.X_test
        restored_object_zphot.model_index = model_index_zphot
        restored_object_zphot.model_path = model_path
        restored_object_zphot.load_ANN(weights_flag)

        dummy_zspec = np.ones(self.X_train_norm.shape[0])
        X_train_zphot = np.zeros((self.X_train_norm.shape[0], self.X_train_norm.shape[1]+2))
        X_train_zphot[:, 0:-2] = self.X_train_norm
        print(self.X_train_norm[0,:])
        print(self.X_train_norm[0, :-1])

        Y_pred_zphot, z_phot_pdf, prediction_report = utils_regression.compute_PDF(restored_object_zphot.normalization, restored_object_zphot.normalizer, restored_object_zphot.data_imputation_values, self.DES_id_train, self.X_train_norm[:, :-1], dummy_zspec, restored_object_zphot.model, 100, 0.9, 1, 0.05)

        X_train_zphot[:, -2] = Y_pred_zphot
        X_train_zphot[:, -1] = prediction_report['stdev']

        self.X_train_norm = X_train_zphot

        dummy_zspec = np.ones(self.X_val_norm.shape[0])
        X_val_zphot = np.zeros((self.X_val_norm.shape[0], self.X_val_norm.shape[1]+2))
        X_val_zphot[:, 0:-2] = self.X_val_norm

        Y_pred_zphot, z_phot_pdf, prediction_report = utils_regression.compute_PDF(restored_object_zphot.normalization, restored_object_zphot.normalizer, restored_object_zphot.data_imputation_values, self.DES_id_val, self.X_val_norm[:, :-1], dummy_zspec, restored_object_zphot.model, 100, 0.9, 1, 0.05)

        X_val_zphot[:, -2] = Y_pred_zphot
        X_val_zphot[:, -1] = prediction_report['stdev']

        self.X_val_norm = X_val_zphot


        return

def evaluate_cv(classifier, model_indexes, weights_flag='final'):
    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    all_Y_pred = []
    classnames = None
    Y_test = None
    DES_id_test = None
    constraints = None
    for model_index in model_indexes:
        model_path = os.path.join(script_dir, 'model_' + classifier + '_classification', str(model_index))
        if classifier == 'ANN':
            with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
                ANN_architecture = json.load(f)
            with open(os.path.join(model_path,'ANN_parameters.json')) as f:
                ANN_parameters = json.load(f)
            with open(os.path.join(model_path,'classification_inputs.json')) as f:
                classification_inputs = json.load(f)

            restored_object = Classification(classifier='ANN',
                                             preprocessing_method=classification_inputs['preprocessing_method'],
                                             catalog_filename=classification_inputs['catalog_filename'],
                                             classification_problem=classification_inputs['classification_problem'],
                                             constraints=classification_inputs['constraints'],
                                             data_imputation=classification_inputs['data_imputation'],
                                             normalization=classification_inputs['normalization'],
                                             train_test_val_split=classification_inputs['train_test_val_split'],
                                             cv_fold_nbr=classification_inputs['cv_fold_nbr'],
                                             cross_validation=classification_inputs['cross_validation'],
                                             tsboard=classification_inputs['tsboard'],
                                             early_stop=classification_inputs['early_stop'],
                                             lrdecay=classification_inputs['lrdecay'],
                                             compute_conf_matr=classification_inputs['compute_conf_matr'],
                                             model_index = model_index,
                                             model_path = model_path
                                            )
            restored_object.load_dataset()
            restored_object.Y_train, restored_object.Y_val, restored_object.Y_test = utils_classification.one_hot_encode(restored_object.Y_train, restored_object.Y_val, restored_object.Y_test)
            restored_object.load_ANN(weights_flag)

            all_Y_pred.append(restored_object.model.predict(restored_object.X_test_norm))

            constraints = restored_object.constraints
            Y_test = restored_object.Y_test
            DES_id_test = restored_object.DES_id_test
            classnames = restored_object.classnames

    mean_Y_pred = np.zeros((Y_test.shape[0], len(classnames)))

    for i in range(Y_test.shape[0]):
        for j in range(len(classnames)):
            for k in range(len(model_indexes)):
                mean_Y_pred[i,j] += (all_Y_pred[k][i][j])
            mean_Y_pred[i,j] = mean_Y_pred[i,j]/len(model_indexes)

    savepath = os.path.join(script_dir, 'model_' + classifier + '_classification', 'testing_cv', constraints)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    mean_auc_roc, mean_auc_pr = utils_classification.compute_aucs(mean_Y_pred, Y_test, DES_id_test, classnames, savepath=savepath, plot=True)

    return

def evaluate_test(classifier, model_index, weights_flag='final'):

    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    model_path = os.path.join(script_dir, 'model_' + classifier + '_classification', str(model_index))

    with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
        ANN_architecture = json.load(f)
    with open(os.path.join(model_path,'ANN_parameters.json')) as f:
        ANN_parameters = json.load(f)
    with open(os.path.join(model_path,'classification_inputs.json')) as f:
        classification_inputs = json.load(f)

    restored_object = Classification(classifier='ANN',
                                     preprocessing_method=classification_inputs['preprocessing_method'],
                                     catalog_filename=classification_inputs['catalog_filename'],
                                     classification_problem=classification_inputs['classification_problem'],
                                     constraints=classification_inputs['constraints'],
                                     data_imputation=classification_inputs['data_imputation'],
                                     normalization=classification_inputs['normalization'],
                                     train_test_val_split=classification_inputs['train_test_val_split'],
                                     cv_fold_nbr=classification_inputs['cv_fold_nbr'],
                                     cross_validation=classification_inputs['cross_validation'],
                                     tsboard=classification_inputs['tsboard'],
                                     early_stop=classification_inputs['early_stop'],
                                     lrdecay=classification_inputs['lrdecay'],
                                     compute_conf_matr=classification_inputs['compute_conf_matr'],
                                     model_index = model_index,
                                     model_path = model_path
                                    )
    restored_object.load_dataset()
    restored_object.Y_train, restored_object.Y_val, restored_object.Y_test = utils_classification.one_hot_encode(restored_object.Y_train, restored_object.Y_val, restored_object.Y_test)
    restored_object.load_ANN(weights_flag)
    Y_pred = restored_object.model.predict(restored_object.X_test_norm)

    savepath = os.path.join(restored_object.model_path, 'performances', 'test')

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    mean_auc_roc, mean_auc_pr = utils_classification.compute_aucs(Y_pred, restored_object.Y_test, restored_object.DES_id_test, restored_object.classnames, savepath=savepath, plot=True)

def evaluate_val(classifier, model_index, weights_flag='final'):

    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    model_path = os.path.join(script_dir, 'model_' + classifier + '_classification', str(model_index))

    with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
        ANN_architecture = json.load(f)
    with open(os.path.join(model_path,'ANN_parameters.json')) as f:
        ANN_parameters = json.load(f)
    with open(os.path.join(model_path,'classification_inputs.json')) as f:
        classification_inputs = json.load(f)

    restored_object = Classification(classifier='ANN',
                                     preprocessing_method=classification_inputs['preprocessing_method'],
                                     catalog_filename=classification_inputs['catalog_filename'],
                                     classification_problem=classification_inputs['classification_problem'],
                                     constraints=classification_inputs['constraints'],
                                     data_imputation=classification_inputs['data_imputation'],
                                     normalization=classification_inputs['normalization'],
                                     train_test_val_split=classification_inputs['train_test_val_split'],
                                     cv_fold_nbr=classification_inputs['cv_fold_nbr'],
                                     cross_validation=classification_inputs['cross_validation'],
                                     tsboard=classification_inputs['tsboard'],
                                     early_stop=classification_inputs['early_stop'],
                                     lrdecay=classification_inputs['lrdecay'],
                                     compute_conf_matr=classification_inputs['compute_conf_matr'],
                                     model_index = model_index,
                                     model_path = model_path
                                    )
    restored_object.load_dataset()
    restored_object.Y_train, restored_object.Y_val, restored_object.Y_test = utils_classification.one_hot_encode(restored_object.Y_train, restored_object.Y_val, restored_object.Y_test)
    restored_object.load_ANN(weights_flag)
    Y_pred = restored_object.model.predict(restored_object.X_val_norm)
    Y_pred_argmax = []
    Y_true_argmax = []
    for i in range(Y_pred.shape[0]):
        Y_pred_argmax.append(np.argmax(Y_pred[i,:]))
        Y_true_argmax.append(np.argmax(restored_object.Y_val[i,:]))

    savepath = os.path.join(restored_object.model_path, 'performances', 'val')

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    prediction_report = {'DES_id': restored_object.DES_id_val, 'Y_true': Y_true_argmax, 'Y_pred': Y_pred_argmax, 'Others_prob': Y_pred[:,0], 'BG_prob': Y_pred[:,1], 'LRG_prob': Y_pred[:,2], 'ELG_prob': Y_pred[:,3], 'QSO_prob': Y_pred[:,4]}
    pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(savepath, 'Probabilities.csv'), index=False)

    mean_auc_roc, mean_auc_pr = utils_classification.compute_aucs(Y_pred, restored_object.Y_test, restored_object.DES_id_test, restored_object.classnames, savepath=savepath, plot=True)

    return

def evaluate_colors_zphot(classifier, regressor, model_index_colors, model_index_zphot, threshold_colors, weights_flag='final'):

    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    classnames_colors = []
    X_test = None
    Y_test = None
    DES_id_test = None
    constraints = None
    model_path_colors = os.path.join(script_dir, 'model_' + classifier + '_classification', str(model_index_colors))

    if classifier == 'ANN':
        with open(os.path.join(model_path_colors, 'ANN_architecture.json')) as f:
            ANN_architecture = json.load(f)
        with open(os.path.join(model_path_colors,'ANN_parameters.json')) as f:
            ANN_parameters = json.load(f)
        with open(os.path.join(model_path_colors,'classification_inputs.json')) as f:
            classification_inputs = json.load(f)

        restored_object_colors = Classification(classifier='ANN',
                                         preprocessing_method=classification_inputs['preprocessing_method'],
                                         catalog_filename=classification_inputs['catalog_filename'],
                                         classification_problem=classification_inputs['classification_problem'],
                                         constraints=classification_inputs['constraints'],
                                         data_imputation=classification_inputs['data_imputation'],
                                         normalization=classification_inputs['normalization'],
                                         train_test_val_split=classification_inputs['train_test_val_split'],
                                         cv_fold_nbr=classification_inputs['cv_fold_nbr'],
                                         cross_validation=classification_inputs['cross_validation'],
                                         tsboard=classification_inputs['tsboard'],
                                         early_stop=classification_inputs['early_stop'],
                                         lrdecay=classification_inputs['lrdecay'],
                                         compute_conf_matr=classification_inputs['compute_conf_matr']
                                        )
        restored_object_colors.load_dataset()
        restored_object_colors.Y_train, restored_object_colors.Y_val, restored_object_colors.Y_test = utils_classification.one_hot_encode(restored_object_colors.Y_train, restored_object_colors.Y_val, restored_object_colors.Y_test)
        restored_object_colors.model_index = model_index_colors
        restored_object_colors.model_path = model_path_colors
        restored_object_colors.load_ANN(weights_flag)
        Y_pred_colors = restored_object_colors.model.predict(restored_object_colors.X_val_norm)

    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    model_path = os.path.join(script_dir, 'model_' + regressor + '_regression', str(model_index_zphot))

    if regressor == 'ANN':
        with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
            ANN_architecture = json.load(f)
        with open(os.path.join(model_path,'ANN_parameters.json')) as f:
            ANN_parameters = json.load(f)
        with open(os.path.join(model_path,'regression_inputs.json')) as f:
            regression_inputs = json.load(f)

        restored_object_zphot = Regression(regressor='ANN',
                                         preprocessing_method=regression_inputs['preprocessing_method'],
                                         catalog_filename=regression_inputs['catalog_filename'],
                                         regression_problem=regression_inputs['regression_problem'],
                                         constraints=regression_inputs['constraints'],
                                         data_imputation=regression_inputs['data_imputation'],
                                         normalization=regression_inputs['normalization'],
                                         train_test_val_split=regression_inputs['train_test_val_split'],
                                         cv_fold_nbr=regression_inputs['cv_fold_nbr'],
                                         cross_validation=regression_inputs['cross_validation'],
                                         tsboard=regression_inputs['tsboard'],
                                         early_stop=regression_inputs['early_stop'],
                                         lrdecay=regression_inputs['lrdecay'],
                                         zphot_conf_threshold=regression_inputs['zphot_conf_threshold']
                                        )
        restored_object_zphot.load_dataset()
        restored_object_zphot.model_index = model_index_zphot
        restored_object_zphot.model_path = model_path
        restored_object_zphot.load_ANN(weights_flag)

        # X_val_color_processed = utils_regression.fill_empty_entries(restored_object_colors.X_val[:, :-1], restored_object_zphot.data_imputation_values)
        # X_val_color_processed = utils_regression.apply_normalization(X_test_color_processed, restored_object_zphot.normalization, restored_object_zphot.normalizer)
        # Y_pred_zphot = restored_object_zphot.model.predict(X_test_color_processed)
        # Y_pred_zphot = np.squeeze(Y_pred_zphot)

        Y_pred_zphot, z_phot_pdf, prediction_report = utils_regression.compute_PDF(restored_object_zphot.normalization, restored_object_zphot.normalizer, restored_object_zphot.data_imputation_values, restored_object_colors.DES_id_val, restored_object_colors.X_val[:, :-1], dummy_zspec, restored_object_zphot.model, 100, 0.9, 1, 0.05)

        z_phot_pdf_dict = {'DES_id': [], 'bin_min': [], 'bin_max': [], 'probability': [], 'z_peak': [], 'z_spec': []}

        for i in range(restored_object_colors.DES_id_val.shape[0]):
            all_bin_min = []
            all_bin_max = []
            all_prob = []
            z_phot_pdf_dict['DES_id'].append(restored_object_colors.DES_id_val[i])
            z_phot_pdf_dict['z_peak'].append(Y_pred_zphot[i])
            z_phot_pdf_dict['z_spec'].append(dummy_zspec[i])
            for j in range(len(z_phot_pdf[i])):
                all_bin_min.append(z_phot_pdf[i][j][0])
                all_bin_max.append(z_phot_pdf[i][j][1])
                all_prob.append(z_phot_pdf[i][j][2])
            z_phot_pdf_dict['bin_min'].append(all_bin_min)
            z_phot_pdf_dict['bin_max'].append(all_bin_max)
            z_phot_pdf_dict['probability'].append(all_prob)

    # conf_mask_colors = np.where(np.amax(Y_pred_colors, axis=1) < threshold_colors)[0]
    # for i in conf_mask_colors:
    #     if np.argmax(Y_pred_colors[i, :]) == 1.0:
    #         if (Y_pred_zphot[i, 0] < 0.05) or (Y_pred_zphot[i, 0] > 0.4):
    #             if (Y_pred_zphot[i, 2] < 0.0005):
    #                 Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
    #             else:
    #                 continue
    #     elif np.argmax(Y_pred_colors[i, :]) == 2.0 :
    #         if (Y_pred_zphot[i, 0] < 0.4) or (Y_pred_zphot[i, 0] > 0.75):
    #             if (Y_pred_zphot[i, 2] < 0.005):
    #                 Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
    #             else:
    #                 continue
    #     elif np.argmax(Y_pred_colors[i, :]) == 3.0:
    #         if (Y_pred_zphot[i, 0] < 0.75) or (Y_pred_zphot[i, 0] > 1.1):
    #             if (Y_pred_zphot[i, 2] < 0.001):
    #                 Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
    #             else:
    #                 continue
    #     elif np.argmax(Y_pred_colors[i, :]) == 4.0:
    #         if (Y_pred_zphot[i, 0] < 1.1):
    #             if (Y_pred_zphot[i, 2] < 0.002):
    #                 Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
    #             else:
    #                 continue

        # conf_mask_colors = np.where(np.amax(Y_pred_colors, axis=1) < threshold_colors)[0]
        # for i in conf_mask_colors:
        #     if np.argmax(Y_pred_colors[i, :]) == 1.0:
        #         if (z_phot_pdf_dict['bin_max'][i][-1] < 0.05) or (z_phot_pdf_dict['bin_min'][i][0] > 0.4):
        #             Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
        #     elif np.argmax(Y_pred_colors[i, :]) == 2.0 :
        #         if (z_phot_pdf_dict['bin_max'][i][-1] < 0.4) or (z_phot_pdf_dict['bin_min'][i][0] > 0.75):
        #             Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
        #     elif np.argmax(Y_pred_colors[i, :]) == 3.0:
        #         if (z_phot_pdf_dict['bin_max'][i][-1] < 0.75) or (z_phot_pdf_dict['bin_min'][i][0] > 1.1):
        #             Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
        #     elif np.argmax(Y_pred_colors[i, :]) == 4.0:
        #         if (z_phot_pdf_dict['bin_max'][i][-1] < 1.1):
        #             Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]

        conf_mask_colors = np.where(np.amax(Y_pred_colors, axis=1) < threshold_colors)[0]
        probabilities_BG = []
        probabilities_LRG = []
        probabilities_ELG = []
        probabilities_QSO = []
        Y_pred_colors_labels = []
        Y_val_colors_labels = []
        pdf_variance = []

        for i in range(restored_object_colors.DES_id_val.shape[0]):
            Y_pred_colors_labels.append(np.argmax(Y_pred_colors[i, :]))
            Y_val_colors_labels.append(np.argmax(restored_object_colors.Y_val[i, :]))
            probabilities_BG.append(utils_regression.compute_PDF_probability(z_phot_pdf[i], 0.05, 0.4))
            probabilities_LRG.append(utils_regression.compute_PDF_probability(z_phot_pdf[i], 0.4, 0.75))
            probabilities_ELG.append(utils_regression.compute_PDF_probability(z_phot_pdf[i], 0.75, 1.1))
            probabilities_QSO.append(utils_regression.compute_PDF_probability(z_phot_pdf[i], 1.1, 100))
            pdf_variance.append(utils_regression.compute_PDF_variance(z_phot_pdf[i]))
        # for i in conf_mask_colors:
        #     if np.argmax(Y_pred_colors[i, :]) == 1.0:
        #         if (utils_regression.compute_PDF_probability(z_phot_pdf[i], 0.05, 0.4) < 0.5):
        #             Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
        #     elif np.argmax(Y_pred_colors[i, :]) == 2.0 :
        #         if (utils_regression.compute_PDF_probability(z_phot_pdf[i], 0.4, 0.75) < 0.5):
        #             Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
        #     elif np.argmax(Y_pred_colors[i, :]) == 3.0:
        #         if (utils_regression.compute_PDF_probability(z_phot_pdf[i], 0.75, 1.1) < 0.5):
        #             Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
        #     elif np.argmax(Y_pred_colors[i, :]) == 4.0:
        #         if (utils_regression.compute_PDF_probability(z_phot_pdf[i], 1.1, 100) < 0.5):
        #             Y_pred_colors[i, :] = [1.0, 0.0, 0.0, 0.0, 0.0]

    savepath = os.path.join(script_dir, 'model_ANN_colors_zphot', 'colors_' + str(model_index_colors) + 'zphot_' + str(model_index_zphot))

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    prediction_report = {'DES_id': restored_object_colors.DES_id_val, 'Y_true': Y_val_colors_labels, 'Y_pred': Y_pred_colors_labels, 'z_peak': z_phot_pdf_dict['z_peak'], 'pdf_variance': pdf_variance, 'Probabilities_BG': probabilities_BG, 'Probabilities_LRG': probabilities_LRG, 'Probabilities_ELG': probabilities_ELG, 'Probabilities_QSO': probabilities_QSO}
    pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(savepath, 'Probabilities.csv'), index=False)

    mean_auc_roc, mean_auc_pr = utils_classification.compute_aucs(Y_pred_colors, restored_object_colors.Y_val, restored_object_colors.DES_id_val, restored_object_colors.classnames, savepath=savepath, plot=True)

    return

def lr_decay_custom_callback(lr_init, lr_decay, min_lr):
    from keras.callbacks import LearningRateScheduler
    def step_decay(epoch):
        new_lr = lr_init*(lr_decay**(epoch + 1))
        if new_lr > min_lr:
            return new_lr
        else:
            return min_lr
    return LearningRateScheduler(step_decay)

def f1(y_true, y_pred):

    import keras.backend as K
    import tensorflow as tf

    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

if __name__ == '__main__':

    evaluate_val('ANN', 149)
    # evaluate_colors_zphot('ANN', 'ANN', 149, 100, 1.0, weights_flag='best')

    # Default ANN run without preprocessing

    # BG_ELG_LRG_QSO_classification = Classification( lrdecay=True, early_stop=True, compute_conf_matr=True)
    # BG_ELG_LRG_QSO_classification.load_dataset()
    # BG_ELG_LRG_QSO_classification.run()

    # ANN run with preprocessing

    # preprocessing_method = [{'method': 'ENN_us', 'arguments':['majority', 6, 'all']}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=0.0, normalization=True, lrdecay=True, early_stop=True, compute_conf_matr=True)
    # BG_ELG_LRG_QSO_classification.load_dataset()
    # BG_ELG_LRG_QSO_classification.run()

    # ANN gridsearch

    # preprocessing_method = [{'method': 'RANDOM_us', 'arguments':[150]}, {'method': 'Borderline_SMOTE_os', 'arguments':[2, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='max', constraints='nooutcolors+mag', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

    # preprocessing_method = [{'method': 'RANDOM_us', 'arguments':[150]}, {'method': 'ADASYN_os', 'arguments':[2, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='max', constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='max', constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

    preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4, 5]}]
    BG_ELG_LRG_QSO_classification = Classification(ANN_zphot=True, preprocessing_method=preprocessing_method, data_imputation='max', constraints='nooutcrossover', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'RANDOM_os', 'arguments':[4]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='max', constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='mean', constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'RANDOM_os', 'arguments':[4]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='mean', constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[2, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='max', constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=3)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[2, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='max', constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=4)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[2, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation='max', constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=5)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

    # evaluate_cv('ANN', [49, 50, 51, 52, 53])

    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=-1, constraints='nooutcolors', normalization='quant', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=0.0, constraints='noout', normalization='cr', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4, 5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=-1, constraints='noout', normalization='quant', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4,5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=-1.0, constraints='noout', normalization='quant', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4,5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=-1.0, constraints='noout', normalization='quant', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=2)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4,5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=-1.0, constraints='noout', normalization='quant', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=3)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4,5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=-1.0, constraints='noout', normalization='quant', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=4)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()
    #
    # preprocessing_method = [{'method': 'ADASYN_os', 'arguments':[4,5]}]
    # BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method, data_imputation=-1.0, constraints='noout', normalization='quant', lrdecay=True, early_stop=True, compute_conf_matr=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=5)
    # BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

    #Random Forest gridsearch

    # BG_ELG_LRG_QSO_classification = Classification(classifier='RF')
    # BG_ELG_LRG_QSO_classification.load_dataset()
    # BG_ELG_LRG_QSO_classification.init_RF()
    # param_dist = {'n_estimators' : [10, 20, 30]                                   #Number of trees, normally the higher, the better
    #               'max_depth': [2, 3, 4],                                         #Depth of forest should not be too high for noisy data
    #               'bootstrap': [True, False],                                     #Use bottstrap
    #               'C': [0.1, 1, 10, 100],                                         #Number of features to test at each node
    #               'class_weight': [None, 'balanced']}                             #Metric used for evaluation
    # cv_metric  ='f1-score'
    # BG_ELG_LRG_QSO_classification.CV_gridsearch(params_dist, cv_metric)

    #Support Vector Machine gridsearch

    # BG_ELG_LRG_QSO_classification = Classification(classifier='SVM')
    # BG_ELG_LRG_QSO_classification.load_dataset()
    # BG_ELG_LRG_QSO_classification.init_SVM()
    # param_dist = {'kernel' : ['poly', 'rbf', sigmoid]                                    #Number of trees, normally the higher, the better
    #               'degree': [3, 4, 5],                                   #Depth of forest should not be too high for noisy data
    #               'bootstrap': [True, False],                               #Use bottstrap
    #               'max_features': ['auto', 'log2', 8],                      #Number of features to test at each node
    #               'criterion': ['gini', 'entropy']}                         #Metric used for evaluation
    # cv_metric  ='f1-score'
    # BG_ELG_LRG_QSO_classification.CV_gridsearch(params_dist, cv_metric)

    # To compute all preprocessed_dataset

    # preprocessing_methods = [[{'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}],
    #                          [{'method': 'SMOTE_Tomek', 'arguments':['auto']}],
    #                          [{'method': 'SMOTE_os', 'arguments':['auto', 5]}],
    #                          [{'method': 'ADASYN_os', 'arguments':['auto', 5]}],
    #                          [{'method': 'RANDOM_os', 'arguments':['auto']}],
    #                          [{'method': 'ENN_us', 'arguments':['majority', 6, 'all']}],
    #                          [{'method': 'Allknn_us', 'arguments':['majority', 6, 'all']}],
    #                          [{'method': 'Tomek_us', 'arguments':['majority']}],
    #                          [{'method': 'RANDOM_us', 'arguments':['majority']}],
    #                          [{'method': 'CENTROID_us', 'arguments':['majority']}],
    #                          [{'method': 'NearMiss_us', 'arguments':['majority', 6, 6, 1]}],
    #                          [{'method': 'NearMiss_us', 'arguments':['majority', 6, 6, 2]}],
    #                          [{'method': 'NearMiss_us', 'arguments':['majority', 6, 6, 3]}],
    #                          [{'method': 'IHT_us', 'arguments':['majority', 'adaboost', 5]}],
    #                          [{'method': 'ENN_us', 'arguments':['majority', 6, 'all']}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}],
    #                          [{'method': 'NearMiss_us', 'arguments':['majority', 6, 6, 3]}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}],
    #                          [{'method': 'NearMiss_us', 'arguments':['majority', 6, 6, 1]}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}]]

    # for i in preprocessing_methods:
    #     BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=i, early_stop=True, lrdecay=True)
    #     BG_ELG_LRG_QSO_classification.load_dataset()


    # def load_dataset(self):
    #
    #     self.DES_id_train, self.X_train, self.Y_train, self.sample_weights_train, self.DES_id_val, self.X_val, self.Y_val, self.sample_weights_val, self.DES_id_test, self.X_test, self.Y_test, self.sample_weights_test, self.data_keys = utils_classification.load_dataset(self.dataset_path, self.model_path, self.classification_problem, self.train_test_val_split, self.cv_fold_nbr, self.normalization, self.data_imputation, self.random_seed)
    #     self.X_train_norm, self.X_val_norm, self.X_test_norm, self.data_imputation_values = utils_classification.fill_empty_entries_dataset(self.X_train, self.X_val,self. X_test, self.data_imputation, self.constraints)
    #     self.X_train_norm, self.X_val_norm, self.X_test_norm, self.normalizer = utils_classification.normalize_dataset(self.X_train_norm, self.X_val_norm, self.X_test_norm, self.normalization, self.random_seed)
    #     if self.preprocessing_method is not None:
    #         already_applied, to_apply = self.check_existing_preprocessed_datasets()
    #         if already_applied:
    #             print('Loading following preprocessed dataset : ', already_applied)
    #             self.load_preprocessed_dataset(already_applied)
    #         if to_apply:
    #             print('Processing dataset with : ', to_apply)
    #             for i in to_apply:
    #                 if len(i['arguments']) == 1:
    #                     self.X_train_norm, self.Y_train = getattr(preprocessing_classification, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0])
    #                 elif len(i['arguments']) == 2:
    #                     self.X_train_norm, self.Y_train = getattr(preprocessing_classification, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1])
    #                 elif len(i['arguments']) == 3:
    #                     self.X_train_norm, self.Y_train = getattr(preprocessing_classification, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1], i['arguments'][2])
    #                 elif len(i['arguments']) == 4:
    #                     self.X_train_norm, self.Y_train = getattr(preprocessing_classification, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1], i['arguments'][2], i['arguments'][3])
    #             preprocessing_classification.save_preprocessed_dataset(self.script_dir, self.catalog_filename, self.constraints, self.data_imputation, self.normalization, self.classification_problem, self.train_test_val_split, self.cv_fold_nbr, self.preprocessing_method, self.X_train_norm, self.Y_train)
    #
    #     self.input_dimensions = self.X_train_norm.shape[1]
    #     self.nbr_classes = np.unique(self.Y_test).shape[0]
    #
    #     return
