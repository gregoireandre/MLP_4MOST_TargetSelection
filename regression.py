import __init__
import preprocessing_regression
from utils_regression import *

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


class Regression():

    def __init__(self, regressor='ANN', preprocessing_method=None, random_seed=7, catalog_filename='4MOST.CatForGregoire.05Dec2018.zphot', regression_problem='zphot_regression', constraints='no', data_imputation=0.0, normalization=None, train_test_val_split=[80, 20, 20], cv_fold_nbr=1, zphot_conf_threshold=0.0, zphot_risk_threshold=1.0, zphot_safe_threshold=0.0, cross_validation=False, tsboard=False, early_stop=True, lrdecay=True, model_index=None, model_path=None):

        self.regressor = regressor                                            # regressor (str) : 'ANN', 'RF', 'SVM'
        self.preprocessing_method = preprocessing_method                        # Processing method (dict) : {'method' : 'method to use among preprocessing.py', 'arguments': [arg1, arg2, arg3, ...]}
        self.random_seed = random_seed                                          # Random seed (int) : Random seed to use for reproducibility
        self.catalog_filename = catalog_filename                                # Catalog Filename (str) : Filename of the catalog to use (must be a .fits and located in "src" folder)
        self.regression_problem = regression_problem                    # regression problem (str) : regression problem to consider (see dataset_generator.py for details about different regression problem)
        self.constraints= constraints                                           # Constraints (str or dict) : Constraints to use on the dataset (see dataset_generator.py for details about different regression problem)
        self.data_imputation = data_imputation                                  # Data imputation (float) : Value used while generating the dataset to fill empty entries
        self.normalization = normalization
        self.train_test_val_split = train_test_val_split                    # Train Test split (list) : Fraction of dataset to use on training/testing and on training/validation (e.g [80, 20])
        self.zphot_conf_threshold = zphot_conf_threshold
        self.zphot_risk_threshold = zphot_risk_threshold
        self.zphot_safe_threshold = zphot_safe_threshold
        self.cv_fold_nbr = cv_fold_nbr                                          # Cross Validation fold number (int) : The index of the cross validation dataset to use (refer to dataset_generator.py for more info)
        self.cross_validation = cross_validation                                # Cross Validation (bool) : Weither or not use cross validation during evaluation of the model
        self.tsboard = tsboard                                                  # Tensorboard (bool) : Weither or not use tensorboard callback during training of the model
        self.early_stop = early_stop                                            # Early Stop (bool) : Weither or not use early stop callback during training of the model
        self.lrdecay = lrdecay                                                  # Learning Rate decay (bool) : Weither or not use learning rate plateau decay callback during training of the model
        self.model_index = model_index
        self.model_path = model_path

        np.random.seed(self.random_seed)                                        # Fix Numpy random seed for reproducibility
        self.script_path = os.path.realpath(__file__)
        self.script_dir, _ = os.path.split(self.script_path)

        self.input_dict = {
                            'regressor': regressor,
                            'preprocessing_method': preprocessing_method,
                            'random_seed': random_seed,
                            'catalog_filename': catalog_filename,
                            'regression_problem': regression_problem,
                            'constraints': constraints,
                            'data_imputation': data_imputation,
                            'normalization': normalization,
                            'train_test_val_split': train_test_val_split,
                            'zphot_safe_threshold': zphot_safe_threshold,
                            'cv_fold_nbr': cv_fold_nbr,
                            'cross_validation': cross_validation,
                            'tsboard': tsboard,
                            'early_stop': early_stop,
                            'lrdecay': lrdecay,
                            'random_seed': random_seed,
                            'constraints': constraints
                          }

        self.dataset_path = os.path.join(self.script_dir, 'datasets', self.catalog_filename, self.constraints + '-constraints')

        # Create model_ANN folder in "src" directory

        if not os.path.exists(os.path.join(self.script_dir, 'model_' + self.regressor +'_regression')):
            os.makedirs(os.path.join(self.script_dir, 'model_' + self.regressor +'_regression'))

        # To allow effective comparison of increasing number of model, the models are named iteratively with an integer beginning at 1.
        # A folder named after the model is then created and all the information realtive to the model and its performance is stored in the latter.

        if model_index is None:
            # Compute model filename by counting the number of folder in the "model_ANN" directory
            nbr_directories = len(next(os.walk(os.path.join(self.script_dir, 'model_ANN_regression')))[1])
            self.model_index = nbr_directories+1

        if model_path is None:
            self.model_path = os.path.join(self.script_dir, 'model_ANN_regression', str(self.model_index))
            os.makedirs(self.model_path)

        print('Initialized, will load dataset')

    def run(self):

        if self.regressor == 'ANN':
            self.define_ANN()
            self.init_ANN()
            self.train_ANN()
            self.evaluate_ANN()

        return

    def load_dataset(self):
        self.DES_id_train, self.X_train, self.Y_train, self.sample_weights_train, self.DES_id_val, self.X_val, self.Y_val, self.sample_weights_val, self.DES_id_test, self.X_test, self.Y_test, self.sample_weights_test, self.data_keys = load_dataset(self.dataset_path, self.model_path, self.regression_problem, self.zphot_safe_threshold, self.train_test_val_split, self.cv_fold_nbr, self.normalization, self.data_imputation, self.random_seed)
        self.X_train_norm, self.X_val_norm, self.X_test_norm, self.normalizer = normalize_dataset(self.X_train, self.X_val,self. X_test, self.normalization, self.random_seed)
        self.X_train_norm, self.X_val_norm, self.X_test_norm, self.data_imputation_values = fill_empty_entries_dataset(self.X_train_norm, self.X_val_norm,self. X_test_norm, self.data_imputation, self.constraints)
        # self.X_train_norm, self.X_val_norm, self.X_test_norm, self.data_imputation_values = fill_empty_entries_dataset(self.X_train, self.X_val,self. X_test, self.data_imputation, self.constraints)
        # self.X_train_norm, self.X_val_norm, self.X_test_norm, self.normalizer = normalize_dataset(self.X_train_norm, self.X_val_norm, self.X_test_norm, self.normalization, self.random_seed)
        if self.preprocessing_method is not None:
            already_applied, to_apply = self.check_existing_preprocessed_datasets()
            if already_applied:
                print('Loading following preprocessed dataset : ', already_applied)
                self.load_preprocessed_dataset(already_applied)
            if to_apply:
                print('Processing dataset with : ', to_apply)
                for i in to_apply:
                    if len(i['arguments']) == 1:
                        self.X_train_norm, self.Y_train = getattr(preprocessing, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0])
                    elif len(i['arguments']) == 2:
                        self.X_train_norm, self.Y_train = getattr(preprocessing, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1])
                    elif len(i['arguments']) == 3:
                        self.X_train_norm, self.Y_train = getattr(preprocessing, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1], i['arguments'][2])
                    elif len(i['arguments']) == 4:
                        self.X_train_norm, self.Y_train = getattr(preprocessing, i['method'])(self.X_train_norm, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1], i['arguments'][2], i['arguments'][3])
                preprocessing.save_preprocessed_dataset(self.script_dir, self.catalog_filename, self.constraints, self.data_imputation, self.normalization, self.regression_problem, self.train_test_val_split, self.cv_fold_nbr, self.preprocessing_method, self.X_train_norm, self.Y_train)

        self.input_dimensions = self.X_train_norm.shape[1]
        self.nbr_classes = np.unique(self.Y_test).shape[0]

        return

    def check_existing_preprocessed_datasets(self):

        preprocessed_dataset_path = os.path.join(self.dataset_path, 'preprocessed_' + str(self.data_imputation) + '-imputation', self.regression_problem + '_train_' + str(self.train_test_val_split[0]) + '_' + str(self.train_test_val_split[1]) + '_' + str(self.cv_fold_nbr) + '_norm-' + str(self.normalization))
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

        preprocessed_dataset_path = os.path.join(self.dataset_path, 'preprocessed_' + str(self.data_imputation) + '-imputation', self.regression_problem + '_train_' + str(self.train_test_val_split[0]) + '_' + str(self.train_test_val_split[1]) + '_' + str(self.dataset_idx) + '_' + str(self.cv_fold_nbr) + '_norm-' + str(self.normalization))
        for idx, i in enumerate(preprocessing_methods):
            if idx == 0:
                preprocessed_dataset_filename = i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])
            else:
                preprocessed_dataset_filename += '_' + i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])

        self.training_dataset, _ = read_fits(os.path.join(preprocessed_dataset_path, preprocessed_dataset_filename + '.fits'))
        np.random.shuffle(self.training_dataset)

        # split into input (X) and output (Y) variables
        self.X_train_norm = self.training_dataset[:,:-1]
        self.Y_train = self.training_dataset[:,-1]

        self.input_dimensions = self.X_train_norm.shape[1]
        self.nbr_classes = np.unique(self.Y_test).shape[0]

        return

    def run_ANN_gridsearch(self, max_run=250):

        global keras
        import keras
        global set_random_seed
        from tensorflow import set_random_seed
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

        self.ann_parameters_talos =  {'loss_function': ['mean_absolute_error'],
                                      'learning_rate': [0.001],
                                      'batch_size': [256],
                                      'epochs': [400],
                                      'metrics': [['mean_absolute_error', 'mean_squared_error']],
                                      'nbr_layers': [3],
                                      'nbr_neurons' : [30],
                                      'input_activation' : ['tanh'],
                                      'activation' : ['tanh'],
                                      'output_activation': ['linear'],
                                      'dropout_strength': [0.0],
                                      'kernel_initializer': ['lecun_normal'],
                                      'bias_initializer' : ['zeros'],
                                      'kernel_regularizer': [None],
                                      'bias_regularizer': [None],
                                      'activity_regularizer': [None],
                                      'kernel_constraint': [None],
                                      'bias_constraint': [None],
                                      'weighted': [False, True],
                                      'batch_normalization': [False, True]}

        all_keys = list(self.ann_parameters_talos.keys())
        all_combinations = list(itertools.product(*(self.ann_parameters_talos[key] for key in all_keys)))
        print('ANN GridSearch has ' + str(len(all_combinations)) + ' combinations')
        if len(all_combinations) > max_run:
            print('Randomly select ' + str(max_run) + ' among them')
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_run]
        for idx, combination in enumerate(all_combinations):
            if idx > 0:
                nbr_directories = len(next(os.walk(os.path.join(self.script_dir, 'model_' + self.regressor + '_regression')))[1])
                self.model_index = nbr_directories+1
                self.model_path = os.path.join(self.script_dir, 'model_' + self.regressor + '_regression', str(self.model_index))
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
            self.layers_list.append({'size': 1, 'activation': self.ann_parameters['output_activation']})

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
        self.model.compile(loss=self.ann_parameters['loss_function'], optimizer=self.ann_parameters['optimizer'], metrics=self.ann_parameters['metrics'])

        return

    def train_ANN(self):

        from custom_metrics_regression import Metrics

        self.custom_metrics = Metrics()

        # Create model folder architecture

        if not os.path.exists(os.path.join(self.model_path, 'tsboard')):
            os.makedirs(os.path.join(self.model_path, 'tsboard'))

        if not os.path.exists(os.path.join(self.model_path, 'performances', 'val')):
            os.makedirs(os.path.join(self.model_path, 'performances', 'val'))

        if not os.path.exists(os.path.join(self.model_path, 'checkpoints')):
            os.makedirs(os.path.join(self.model_path, 'checkpoints'))

        # Store the ANN parameters in a json file and save them in the model folder

        with open(os.path.join(self.model_path, 'regression_inputs.json'), 'w') as fp:
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
            self.history = self.model.fit(self.X_train_norm, self.Y_train, sample_weight=self.sample_weights_train, validation_data=(self.X_val_norm, self.Y_val, self.sample_weights_val), epochs=self.ann_parameters['epochs'], batch_size=self.ann_parameters['batch_size'], callbacks=callbacks_list, verbose=2)
        else:
            self.history = self.model.fit(self.X_train_norm, self.Y_train, validation_data=(self.X_val_norm, self.Y_val), epochs=self.ann_parameters['epochs'], batch_size=self.ann_parameters['batch_size'], callbacks=callbacks_list, verbose=2)

        last_MAE = self.history.history['val_mean_absolute_error'][-1]
        self.score = last_MAE
        best_MAE = max(self.history.history['val_mean_absolute_error'])
        best_MAE_epoch = self.history.history['val_mean_absolute_error'].index(best_MAE) + 1

        directories = os.listdir(os.path.join(self.model_path, 'checkpoints'))
        for idx, i in enumerate(directories):
            if '{:03}'.format(best_MAE_epoch) in i:
                new_filename = i.split('.')[0] + '_' + str(best_MAE_epoch) + '-MAE.hdf5'
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
            self.model.save_weights(os.path.join(self.model_path, 'ANN_weights_' + str(self.early_stopped_epoch) + '-epo_' + str(last_MAE) + '-MAE.hdf5'))
        else:
            self.model.save_weights(os.path.join(self.model_path, 'ANN_weights_' + str(self.ann_parameters['epochs']) + '-epo_' + str(last_MAE) + '-MAE.hdf5'))

        print("Saved model to disk")

        # loss
        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'ANN_loss.png'))

        plt.close()

        # loss
        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.history.history['mean_absolute_error'])
        plt.plot(self.history.history['val_mean_absolute_error'])
        plt.title('Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'ANN_mae.png'))

        plt.close()

        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.bias)
        plt.plot(self.custom_metrics.epochs, self.custom_metrics.variance)
        plt.title('Bias-Variance Of the Model')
        plt.ylabel('Bias/VarianceSScore')
        plt.xlabel('epoch')
        plt.legend(['bias', 'variance'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, 'performances', 'val', 'bias_var.png'))

        plt.clf()

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
        weights, self.score = get_model_weights_path(self.model_path, weights_flag)
        self.model.load_weights(weights)
        self.model.compile(loss=ANN_parameters['loss_function'], optimizer=Adam(lr=ANN_parameters['learning_rate']), metrics=[])

        return

    def evaluate_ANN(self):

        # evaluate the model
        Y_pred = self.model.predict(self.X_val_norm)
        Y_pred = np.squeeze(Y_pred)

        mae = []

        for i in range(Y_pred.shape[0]):
            mae.append(abs(self.Y_val[i] - Y_pred[i]))

        savepath = os.path.join(self.model_path, 'performances', 'val', 'ANN')

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        prediction_report = {'DES_id': self.DES_id_val, 'Y_true': self.Y_val, 'Y_pred': Y_pred}
        pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(savepath, 'Predictions.csv'), index=False)

        plot_regression_scatterplot(Y_pred, self.Y_val, self.score, savepath)
        plot_histogram_distribution(Y_pred, self.Y_val, self.score, savepath)
        # heatmap_density_plot(Y_pred, self.Y_val, self.score, savepath)

        report2csv(self.model_index, os.path.join(self.script_dir, 'model_' + self.regressor + '_regression'), self.catalog_filename, self.constraints, self.ann_parameters, self.regression_problem, self.zphot_safe_threshold, self.train_test_val_split, self.cv_fold_nbr, self.data_imputation, self.normalization, self.model_path, self.preprocessing_method, self.early_stopped_epoch, self.score)

        # zphot_pdf, HSC_statistics_dict, metaphor_statistics_dict = compute_PDF(self.normalization, self.normalizer, self.data_imputation_values, self.DES_id_val, self.X_val, self.Y_val, self.model, 100, 0.9, 1, 0.05)
        #
        # HSC_statistics_dict['DES_id'] = self.DES_id_val
        # metaphor_statistics_dict['DES_id'] = self.DES_id_val
        #
        # zphot_pdf_dict = {
        #                     'DES_id': [],
        #                     'z_spec': [],
        #                     'bin_min': [],
        #                     'bin_max': [],
        #                     'z_peak': [],
        #                     'probability': [],
        #                     'variance': [],
        #                     'probability_BG': [],
        #                     'probability_LRG': [],
        #                     'probability_ELG': [],
        #                     'probability_QSO': []
        #                    }
        # idx_best_pdf = np.argmax(prediction_report['perc_0.05'])
        # idx_worse_pdf = np.argmin(prediction_report['perc_0.05'])
        #
        # for i in range(self.DES_id_val.shape[0]):
        #     all_bin_min = []
        #     all_bin_max = []
        #     all_prob = []
        #     zphot_pdf_dict['DES_id'].append(self.DES_id_val[i])
        #     zphot_pdf_dict['z_spec'].append(self.Y_val[i])
        #     for j in range(len(zphot_pdf[i])):
        #         all_bin_min.append(zphot_pdf[i][j][0])
        #         all_bin_max.append(zphot_pdf[i][j][1])
        #         all_prob.append(zphot_pdf[i][j][2])
        #     zphot_pdf_dict['bin_min'].append(all_bin_min)
        #     zphot_pdf_dict['bin_max'].append(all_bin_max)
        #     zphot_pdf_dict['z_peak'].append(Y_pred[i])
        #     zphot_pdf_dict['probability'].append(all_prob)
        #     zphot_pdf_dict['variance'].append(compute_PDF_variance(zphot_pdf[i]))
        #     zphot_pdf_dict['probability_BG'].append(compute_PDF_probability(zphot_pdf[i], 0.05, 0.4))
        #     zphot_pdf_dict['probability_LRG'].append(compute_PDF_probability(zphot_pdf[i], 0.4, 0.75))
        #     zphot_pdf_dict['probability_ELG'].append(compute_PDF_probability(zphot_pdf[i], 0.75, 1.1))
        #     zphot_pdf_dict['probability_QSO'].append(compute_PDF_probability(zphot_pdf[i], 1.1, 100))
        #
        # savepath_pdf = os.path.join(self.model_path, 'performances', 'val', 'pdf')
        #
        # if not os.path.exists(savepath_pdf):
        #     os.makedirs(savepath_pdf)
        #
        # pd.DataFrame.from_dict(zphot_pdf_dict).to_csv(os.path.join(savepath_pdf, 'pdfs.csv'), index=False)
        # pd.DataFrame.from_dict(HSC_statistics_dict).to_csv(os.path.join(savepath_pdf, 'pdfs_HSC_statistics.csv'), index=False)
        # pd.DataFrame.from_dict(metaphor_statistics_dict).to_csv(os.path.join(savepath_pdf, 'pdfs_METAPHOR_statistics.csv'), index=False)
        #
        # plot_pdf(zphot_pdf[idx_best_pdf], zphot_pdf_dict['z_peak'][idx_best_pdf], self.Y_val[idx_best_pdf], savepath_pdf, 'max')
        # plot_pdf(zphot_pdf[idx_worse_pdf], zphot_pdf_dict['z_peak'][idx_worse_pdf], self.Y_val[idx_worse_pdf], savepath_pdf, 'min')

def evaluate_val(regressor, model_index, weights_flag='final'):

    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    model_path = os.path.join(script_dir, 'model_' + regressor + '_regression', str(model_index))
    if regressor == 'ANN':
        with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
            ANN_architecture = json.load(f)
        with open(os.path.join(model_path,'ANN_parameters.json')) as f:
            ANN_parameters = json.load(f)
        with open(os.path.join(model_path,'regression_inputs.json')) as f:
            regression_inputs = json.load(f)

        restored_object = Regression(regressor='ANN',
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
                                         zphot_safe_threshold=regression_inputs['zphot_safe_threshold'],
                                         model_index = model_index,
                                         model_path = model_path
                                        )
        restored_object.load_dataset()
        restored_object.load_ANN(weights_flag)

        Y_pred = restored_object.model.predict(restored_object.X_val_norm)
        Y_pred = np.squeeze(Y_pred)

        mae = []

        for i in range(Y_pred.shape[0]):
            mae.append(abs(restored_object.Y_val[i] - Y_pred[i]))

        savepath = os.path.join(restored_object.model_path, 'performances', 'val', 'ANN')

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        prediction_report = {'DES_id': restored_object.DES_id_val, 'Y_true': restored_object.Y_val, 'Y_pred': Y_pred}
        pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(savepath, 'Predictions.csv'), index=False)

        plot_regression_scatterplot(Y_pred, restored_object.Y_val, restored_object.score, savepath)
        plot_histogram_distribution(Y_pred, restored_object.Y_val, restored_object.score, savepath)
        # heatmap_density_plot(Y_pred, restored_object.Y_val, restored_object.score, savepath)

        zphot_pdf, HSC_statistics_dict, metaphor_statistics_dict = compute_PDF(restored_object.normalization, restored_object.normalizer, restored_object.data_imputation_values, restored_object.DES_id_val, restored_object.X_val, restored_object.Y_val, restored_object.model, 100, 0.9, 1, 0.05)

        HSC_statistics_dict['DES_id'] = restored_object.DES_id_val
        metaphor_statistics_dict['DES_id'] = restored_object.DES_id_val
        for i in list(HSC_statistics_dict.keys()):
            print(i, ' : ', len(HSC_statistics_dict[i]))

        zphot_pdf_dict = {
                            'DES_id': [],
                            'z_spec': [],
                            'bin_min': [],
                            'bin_max': [],
                            'z_peak': [],
                            'probability': [],
                            'variance': [],
                            'probability_BG': [],
                            'probability_LRG': [],
                            'probability_ELG': [],
                            'probability_QSO': []
                           }
        idx_best_pdf = np.argmax(metaphor_statistics_dict['perc_0.05'])
        idx_worse_pdf = np.argmin(metaphor_statistics_dict['perc_0.05'])

        for i in range(restored_object.DES_id_val.shape[0]):
            all_bin_min = []
            all_bin_max = []
            all_prob = []
            zphot_pdf_dict['DES_id'].append(restored_object.DES_id_val[i])
            zphot_pdf_dict['z_spec'].append(restored_object.Y_val[i])
            for j in range(len(zphot_pdf[i])):
                all_bin_min.append(zphot_pdf[i][j][0])
                all_bin_max.append(zphot_pdf[i][j][1])
                all_prob.append(zphot_pdf[i][j][2])
            zphot_pdf_dict['bin_min'].append(all_bin_min)
            zphot_pdf_dict['bin_max'].append(all_bin_max)
            zphot_pdf_dict['z_peak'].append(HSC_statistics_dict['z_peak'])
            zphot_pdf_dict['probability'].append(all_prob)
            zphot_pdf_dict['variance'].append(compute_PDF_variance(zphot_pdf[i]))
            zphot_pdf_dict['probability_BG'].append(compute_PDF_probability(zphot_pdf[i], 0.05, 0.4))
            zphot_pdf_dict['probability_LRG'].append(compute_PDF_probability(zphot_pdf[i], 0.4, 0.75))
            zphot_pdf_dict['probability_ELG'].append(compute_PDF_probability(zphot_pdf[i], 0.75, 1.1))
            zphot_pdf_dict['probability_QSO'].append(compute_PDF_probability(zphot_pdf[i], 1.1, 100))

        savepath_pdf = os.path.join(restored_object.model_path, 'performances', 'val', 'pdf')

        if not os.path.exists(savepath_pdf):
            os.makedirs(savepath_pdf)

        pd.DataFrame.from_dict(HSC_statistics_dict).to_csv(os.path.join(savepath_pdf, 'pdfs_HSC_statistics.csv'), index=False)
        pd.DataFrame.from_dict(metaphor_statistics_dict).to_csv(os.path.join(savepath_pdf, 'pdfs_METAPHOR_statistics.csv'), index=False)

        plot_pdf(zphot_pdf[idx_best_pdf], zphot_pdf_dict['z_peak'][idx_best_pdf], restored_object.Y_val[idx_best_pdf], savepath_pdf, 'max')
        plot_pdf(zphot_pdf[idx_worse_pdf], zphot_pdf_dict['z_peak'][idx_worse_pdf], restored_object.Y_val[idx_worse_pdf], savepath_pdf, 'min')

    return

def evaluate_cv(regressor, model_indexes, weights_flag='final'):
    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    all_Y_pred = []
    classnames = None
    Y_test = None
    DES_id_test = None
    constraints = None
    mean_score = None

    for model_index in model_indexes:
        model_path = os.path.join(script_dir, 'model_' + regressor + '_regression', str(model_index))
        if regressor == 'ANN':
            with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
                ANN_architecture = json.load(f)
            with open(os.path.join(model_path,'ANN_parameters.json')) as f:
                ANN_parameters = json.load(f)
            with open(os.path.join(model_path,'regression_inputs.json')) as f:
                regression_inputs = json.load(f)

            restored_object = Regression(regressor='ANN',
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
            restored_object.load_dataset()
            restored_object.model_index = model_index
            restored_object.model_path = model_path
            restored_object.load_ANN(weights_flag)
            mean_score += restored_object.score
            all_Y_pred.append(np.squeeze(restored_object.model.predict(restored_object.X_test_norm)))

            constraints = restored_object.constraints
            Y_test = restored_object.Y_test
            DES_id_test = restored_object.DES_id_test

    mean_Y_pred = np.zeros((Y_test.shape[0], len(classnames)))

    for i in range(Y_test.shape[0]):
            for j in range(len(model_indexes)):
                mean_Y_pred[i] += (all_Y_pred[j][i])
            mean_Y_pred[i] = mean_Y_pred[i]/len(model_indexes)

    savepath = os.path.join(script_dir, 'model_' + classifier + '_regression', 'testing_cv', constraints)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    prediction_report = {'DES_id': DES_id_test, 'Y_true': Y_test, 'Y_pred': mean_Y_pred}
    pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(savepath, 'Predictions.csv'), index=False)

    plot_regression_scatterplot(mean_Y_pred, Y_test, mean_score, savepath)
    plot_histogram_distribution(mean_Y_pred, Y_val, mean_score, savepath)
    heatmap_density_plot(mean_Y_pred, Y_val, mean_score, savepath)

    return

def evaluate_test(regressor, model_index, weights_flag='final'):

    script_path = os.path.realpath(__file__)
    script_dir, _ = os.path.split(script_path)
    model_path = os.path.join(script_dir, 'model_' + regressor + '_regression', str(model_index))
    if regressor == 'ANN':
        with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
            ANN_architecture = json.load(f)
        with open(os.path.join(model_path,'ANN_parameters.json')) as f:
            ANN_parameters = json.load(f)
        with open(os.path.join(model_path,'regression_inputs.json')) as f:
            regression_inputs = json.load(f)

        restored_object = Regression(regressor='ANN',
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
                                         zphot_safe_threshold=regression_inputs['zphot_safe_threshold'],
                                         model_index = model_index,
                                         model_path = model_path
                                        )
        restored_object.load_dataset()
        restored_object.load_ANN(weights_flag)

        Y_pred = restored_object.model.predict(restored_object.X_test_norm)
        Y_pred = np.squeeze(Y_pred)

        mae = []

        for i in range(Y_pred.shape[0]):
            mae.append(abs(restored_object.Y_test[i] - Y_pred[i]))

        savepath = os.path.join(restored_object.model_path, 'performances', 'test', 'ANN')

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        prediction_report = {'DES_id': restored_object.DES_id_test, 'Y_true': restored_object.Y_test, 'Y_pred': Y_pred}
        pd.DataFrame.from_dict(prediction_report).to_csv(os.path.join(savepath, 'Predictions.csv'), index=False)
        plot_regression_scatterplot(Y_pred, restored_object.Y_test, restored_object.score, savepath)
        plot_histogram_distribution(Y_pred, restored_object.Y_test, restored_object.score, savepath)
        # heatmap_density_plot(Y_pred, restored_object.Y_test, restored_object.score, savepath)

        zphot_pdf, HSC_statistics_dict, metaphor_statistics_dict = compute_PDF(restored_object.normalization, restored_object.normalizer, restored_object.data_imputation_values, restored_object.DES_id_test, restored_object.X_test, restored_object.Y_test, restored_object.model, 100, 0.9, 1, 0.05)
        HSC_statistics_dict['DES_id'] = restored_object.DES_id_test
        metaphor_statistics_dict['DES_id'] = restored_object.DES_id_test

        zphot_pdf_dict = {
                            'DES_id': [],
                            'z_spec': [],
                            'bin_min': [],
                            'bin_max': [],
                            'z_peak': [],
                            'probability': [],
                            'variance': [],
                            'probability_BG': [],
                            'probability_LRG': [],
                            'probability_ELG': [],
                            'probability_QSO': []
                           }
        idx_best_pdf = np.argmax(prediction_report['perc_0.05'])
        idx_worse_pdf = np.argmin(prediction_report['perc_0.05'])

        for i in range(restored_object.DES_id_test.shape[0]):
            all_bin_min = []
            all_bin_max = []
            all_prob = []
            zphot_pdf_dict['DES_id'].append(restored_object.DES_id_test[i])
            zphot_pdf_dict['z_spec'].append(restored_object.Y_test[i])
            for j in range(len(zphot_pdf[i])):
                all_bin_min.append(zphot_pdf[i][j][0])
                all_bin_max.append(zphot_pdf[i][j][1])
                all_prob.append(zphot_pdf[i][j][2])
            zphot_pdf_dict['bin_min'].append(all_bin_min)
            zphot_pdf_dict['bin_max'].append(all_bin_max)
            zphot_pdf_dict['z_peak'].append(Y_pred[i])
            zphot_pdf_dict['probability'].append(all_prob)
            zphot_pdf_dict['variance'].append(compute_PDF_variance(zphot_pdf[i]))
            zphot_pdf_dict['probability_BG'].append(compute_PDF_probability(zphot_pdf[i], 0.05, 0.4))
            zphot_pdf_dict['probability_LRG'].append(compute_PDF_probability(zphot_pdf[i], 0.4, 0.75))
            zphot_pdf_dict['probability_ELG'].append(compute_PDF_probability(zphot_pdf[i], 0.75, 1.1))
            zphot_pdf_dict['probability_QSO'].append(compute_PDF_probability(zphot_pdf[i], 1.1, 100))

        savepath_pdf = os.path.join(restored_object.model_path, 'performances', 'test', 'pdf')

        if not os.path.exists(savepath_pdf):
            os.makedirs(savepath_pdf)

        pd.DataFrame.from_dict(zphot_pdf_dict).to_csv(os.path.join(savepath_pdf, 'pdfs.csv'), index=False)
        pd.DataFrame.from_dict(HSC_statistics_dict).to_csv(os.path.join(savepath_pdf, 'pdfs_HSC_statistics.csv'), index=False)
        pd.DataFrame.from_dict(metaphor_statistics_dict).to_csv(os.path.join(savepath_pdf, 'pdfs_METAPHOR_statistics.csv'), index=False)

        plot_pdf(zphot_pdf[idx_best_pdf], zphot_pdf_dict['z_peak'][idx_best_pdf], restored_object.Y_test[idx_best_pdf], savepath_pdf, 'max')
        plot_pdf(zphot_pdf[idx_worse_pdf], zphot_pdf_dict['z_peak'][idx_worse_pdf], restored_object.Y_test[idx_worse_pdf], savepath_pdf, 'min')

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

if __name__ == '__main__':

    evaluate_val('ANN', 1)

    # zphot_regression = Regression(data_imputation='max', regression_problem='zphot_regression_ephor', constraints='noout+bright+nostar+zsafe', normalization='cr', lrdecay=True, early_stop=True, tsboard=False, train_test_val_split=[80, 20, 20], cv_fold_nbr=1, zphot_safe_threshold=0.3)
    # zphot_regression.run_ANN_gridsearch()
