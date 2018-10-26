import __init__
import preprocessing
from utils import *

import os
import sys
import math
import json
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

    def __init__(self, classifier='ANN', preprocessing_method=None, random_seed=7, catalog_filename='4MOST.CatForGregoire.11Oct2018.zphot', classification_problem='BG_ELG_LRG_QSO_classification', others_flag='all', constraints='no', data_imputation=0.0, training_testing_split=[80, 20], dataset_idx=1, cv_fold_nbr=1, cross_validation=False, tsboard=False, early_stop=False, lrdecay=False, compute_conf_matr=False):

        self.classifier = classifier                                            # Classifier (str) : 'ANN', 'RF', 'SVM'
        self.preprocessing_method = preprocessing_method                        # Processing method (dict) : {'method' : 'method to use among preprocessing.py', 'arguments': [arg1, arg2, arg3, ...]}
        self.random_seed = random_seed                                          # Random seed (int) : Random seed to use for reproducibility
        self.catalog_filename = catalog_filename                                # Catalog Filename (str) : Filename of the catalog to use (must be a .fits and located in "src" folder)
        self.classification_problem = classification_problem                    # Classification problem (str) : Classification problem to consider (see dataset_generator.py for details about different classification problem)
        self.constraints= constraints                                           # Constraints (str or dict) : Constraints to use on the dataset (see dataset_generator.py for details about different classification problem)
        self.others_flag = others_flag                                          # Others Flag (str) : Can be either 'no' if we don't want to consider 'others' object or 'all' if we want to consider 'others' object
        self.data_imputation = data_imputation                                  # Data imputation (float) : Value used while generating the dataset to fill empty entries
        self.training_testing_split = training_testing_split                    # Train Test split (list) : Fraction of dataset to use on training/testing and on training/validation (e.g [80, 20])
        self.dataset_idx = dataset_idx                                          # Dataset index (int) : The index of the training dataset to use (refer to dataset_generator.py for more info)
        self.cv_fold_nbr = cv_fold_nbr                                          # Cross Validation fold number (int) : The index of the cross validation dataset to use (refer to dataset_generator.py for more info)
        self.cross_validation = cross_validation                                # Cross Validation (bool) : Weither or not use cross validation during evaluation of the model
        self.tsboard = tsboard                                                  # Tensorboard (bool) : Weither or not use tensorboard callback during training of the model
        self.early_stop = early_stop                                            # Early Stop (bool) : Weither or not use early stop callback during training of the model
        self.lrdecay = lrdecay                                                  # Learning Rate decay (bool) : Weither or not use learning rate plateau decay callback during training of the model
        self.compute_conf_matr = compute_conf_matr                              # Compute Confusion Matrix (bool) : Weither or not compute confusion matrix and save to png during model evalutation

        np.random.seed(self.random_seed)                                        # Fix Numpy random seed for reproducibility
        self.script_path = os.path.realpath(__file__)
        self.script_dir, _ = os.path.split(self.script_path)
        self.constraints = constraints_to_str(self.constraints)                 # Convert constraints input varibale to str
        self.classnames = compute_classnames(self.classification_problem, self.others_flag)

        print('Initialized, will load dataset')

        self.dataset_path = os.path.join(self.script_dir, 'datasets', self.catalog_filename, self.others_flag + '-others_' + self.constraints + '-constraints_' + str(self.data_imputation) + '-imputation')

    def run(self):

        if self.classifier == 'ANN':
            self.define_ANN()
            self.init_ANN()
            self.train_ANN()
            self.evaluate_ANN()

        return

    def load_dataset(self):

        self.X_train, self.Y_train, self.X_val, self.Y_val, self.DES_id_val, self.X_test, self.Y_test, self.DES_id_test = load_dataset(self.dataset_path, self.classification_problem, self.training_testing_split, self.dataset_idx, self.cv_fold_nbr)
        self.sample_weights_val = compute_weights(self.Y_val)
        if self.preprocessing_method is not None:
            print(self.preprocessing_method)
            for i in self.preprocessing_method:
                print(i)
                if len(i['arguments']) == 1:
                    self.X_train, self.Y_train = getattr(preprocessing, i['method'])(self.X_train, self.Y_train, self.random_seed, i['arguments'][0])
                elif len(i['arguments']) == 2:
                    self.X_train, self.Y_train = getattr(preprocessing, i['method'])(self.X_train, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1])
                elif len(i['arguments']) == 3:
                    self.X_train, self.Y_train = getattr(preprocessing, i['method'])(self.X_train, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1], i['arguments'][2])
                elif len(i['arguments']) == 4:
                    self.X_train, self.Y_train = getattr(preprocessing, i['method'])(self.X_train, self.Y_train, self.random_seed, i['arguments'][0], i['arguments'][1], i['arguments'][2], i['arguments'][3])
            preprocessing.save_preprocessed_dataset(self.script_dir, self.catalog_filename, self.others_flag, self.constraints, self.data_imputation, self.classification_problem, self.training_testing_split, self.dataset_idx, self.cv_fold_nbr, self.preprocessing_method, self.X_train, self.Y_train)
        self.sample_weights_train = compute_weights(self.Y_train)

        self.input_dimensions = self.X_train.shape[1]
        self.nbr_classes = np.unique(self.Y_test).shape[0]

        return

    def load_preprocessed_dataset(self):

        _, _, self.X_val, self.Y_val, self.DES_id_val, self.X_test, self.Y_test, self.DES_id_test = load_dataset(self.dataset_path, self.classification_problem, self.training_testing_split, self.dataset_idx, self.cv_fold_nbr)
        self.sample_weights_val = compute_weights(self.Y_val)

        preprocessed_dataset_path = os.path.join(self.script_dir, 'datasets', self.catalog_filename,  self.others_flag + '-others_' + self.constraints + '-constraints', 'preprocessed', self.classification_problem + '_train_' + str(self.training_testing_split[0]) + '_' + str(self.training_testing_split[1]) + '_' + str(self.dataset_idx) + '_' + str(self.cv_fold_nbr))
        for idx, i in enumerate(self.preprocessing_method):
            if idx == 0:
                preprocessed_dataset_filename = i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])
            else:
                preprocessed_dataset_filename += '_' + i['method'] + '_' + '_'.join(str(x) for x in i['arguments'])

        self.training_dataset = read_fits(os.path.join(preprocessed_dataset_path, preprocessed_dataset_filename + '.fits'))
        np.random.shuffle(self.training_dataset)

        # split into input (X) and output (Y) variables
        self.X_train = self.training_dataset[:,:-1]
        self.Y_train = self.training_dataset[:,-1]


        self.sample_weights_train = compute_weights(self.Y_train)
        self.input_dimensions = self.X_train.shape[1]
        self.nbr_classes = np.unique(self.Y_test).shape[0]

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
        global ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
        from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

        set_random_seed(self.random_seed)

        # The parameters of the ANN are stored in a dictionnary

        self.ann_parameters =  {'loss_function': 'categorical_crossentropy',
                                'learning_rate': 0.0001,
                                'batch_size': 128,
                                'epochs': 100,
                                'metrics': ['categorical_accuracy'],
                                'nbr_layers': 4,
                                'nbr_neurons' : 64,
                                'activation' : 'relu',
                                'output_activation': 'softmax',
                                'dropout': False,
                                'dropout_strength': 0.0,
                                'kernel_initializer': 'lecun_normal',
                                'bias_initializer' : 'zeros',
                                'kernel_regularizer': None,
                                'bias_regularizer': None,
                                'activity_regularizer': None,
                                'kernel_constraint': None,
                                'bias_constraint': None,
                                'SNN': False,
                                'weighted': False,
                                'normalize': False}

        self.ann_parameters['optimizer'] = Adam(lr=self.ann_parameters['learning_rate'])
        self.ann_parameters['optimizer_str'] = 'adam'
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
                                      'learning_rate': [0.0001],
                                      'batch_size': [64, 256],
                                      'epochs': [100],
                                      'metrics': [['categorical_accuracy']],
                                      'nbr_layers': [3,5,7],
                                      'nbr_neurons' : [32,64,128],
                                      'activation' : ['relu'],
                                      'output_activation': ['softmax'],
                                      'dropout': [False, True],
                                      'dropout_strength': [0.1],
                                      'kernel_initializer': ['lecun_normal'],
                                      'bias_initializer' : ['zeros'],
                                      'kernel_regularizer': [None, l1(0.1), l2(0.1)],
                                      'bias_regularizer': [None],
                                      'activity_regularizer': [None],
                                      'kernel_constraint': [None],
                                      'bias_constraint': [None],
                                      'SNN': [False],
                                      'weighted': [False, True],
                                      'normalize': [False, True]}

        all_keys = list(self.ann_parameters_talos.keys())
        all_combinations = list(itertools.product(*(self.ann_parameters_talos[key] for key in all_keys)))
        if len(all_combinations) > max_run:
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_run]

        for combination in all_combinations:
            self.ann_parameters = dict(zip(all_keys, combination))
            print(self.ann_parameters)
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

            self.ann_parameters['optimizer'] = Adam(lr=self.ann_parameters['learning_rate'])
            self.ann_parameters['optimizer_str'] = 'Adam'

            # The architecture of the ANN is computed from the ANN parameters dictionnary and stored in a list of layers

            self.layers_list = []
            self.layers_list.append({'size': self.ann_parameters['nbr_neurons'], 'input_dim': self.input_dimensions, 'activation': self.ann_parameters['activation']})
            for i in range(1,self.ann_parameters['nbr_layers'] - 1):
                self.layers_list.append({'size': self.ann_parameters['nbr_neurons'], 'activation' : self.ann_parameters['activation']})
            self.layers_list.append({'size': self.nbr_classes, 'activation': self.ann_parameters['output_activation']})

            self.init_ANN()
            self.train_ANN()
            self.evaluate_ANN()

        return

    def init_ANN(self):

        if self.ann_parameters['SNN']:
            model = Sequential()
            for idx,i in enumerate(self.layers_list):
                if 'input_dim' in i.keys():
                    model.add(Dense(i['size'], input_dim=i['input_dim'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['dropout']:
                        model.add(AlphaDropout(self.ann_parameters['dropout_strength']))
                    if self.ann_parameters['normalize']:
                        model.add(BatchNormalization())
                elif(idx == len(self.layers_list) - 1):
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                else:
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['dropout']:
                        model.add(AlphaDropout(self.ann_parameters['dropout_strength']))
                    if self.ann_parameters['normalize']:
                        model.add(BatchNormalization())
        else:
            model = Sequential()
            for idx,i in enumerate(self.layers_list):
                if 'input_dim' in i.keys():
                    model.add(Dense(i['size'], input_dim=i['input_dim'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['dropout']:
                        model.add(Dropout(self.ann_parameters['dropout_strength'], None, self.random_seed))
                    if self.ann_parameters['normalize']:
                        model.add(BatchNormalization())
                elif(idx == len(self.layers_list) - 1):
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                else:
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['dropout']:
                        model.add(Dropout(self.ann_parameters['dropout_strength'], None, self.random_seed))
                    if self.ann_parameters['normalize']:
                        model.add(BatchNormalization())

        self.model = model
        self.model.compile(loss=self.ann_parameters['loss_function'], optimizer=self.ann_parameters['optimizer'], metrics=self.ann_parameters['metrics'])

        return

    def train_ANN(self):

        # Import custom metrics from custom_metrics.py

        from custom_metrics import Metrics

        # Create model_ANN folder in "src" directory

        if not os.path.exists(os.path.join(self.script_dir, 'model_ANN')):
            os.makedirs(os.path.join(self.script_dir, 'model_ANN'))

        # To allow effective comparison of increasing number of model, the models are named iteratively with an integer beginning at 1.
        # A folder named after the model is then created and all the information realtive to the model and its performance is stored in the latter.

        # Compute model filename by counting the number of folder in the "model_ANN" directory
        nbr_directories = len(next(os.walk(os.path.join(self.script_dir, 'model_ANN')))[1])
        self.model_index = nbr_directories+1
        self.model_path = os.path.join(self.script_dir, 'model_ANN', str(self.model_index))

        # Create model folder architecture

        os.makedirs(self.model_path)

        if not os.path.exists(os.path.join(self.model_path, 'tsboard')):
            os.makedirs(os.path.join(self.model_path, 'tsboard'))

        if not os.path.exists(os.path.join(self.model_path, 'figures')):
            os.makedirs(os.path.join(self.model_path, 'figures'))

        if not os.path.exists(os.path.join(self.model_path, 'checkpoints')):
            os.makedirs(os.path.join(self.model_path, 'checkpoints'))

        # One hot encoding of labels to match the use of softmax activation function in the last layer and categorical crossentropy loss function
        # One hot encoding means that if our label can take 5 different values (e.g 0,1,2,3,4) then the labels are transformed in vectors in 5 dimensional space
        # Example, a label of 0 is equivalent to [1, 0, 0, 0, 0] in one hot representation
        #          a label of 3 is equivalent to [0, 0, 0, 1, 0] in one hot representation

        self.Y_train, self.Y_val, self.Y_test = one_hot_encode(self.Y_train, self.Y_val, self.Y_test)
        self.custom_metrics = Metrics()

        # Store the ANN parameters in a json file and save them in the model folder

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
        lrdecay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001)
        # EarlyStopping stops training of the ANN if the quantity monitored does not get better for a given number of epoch (patience)
        earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=15, verbose=1, mode='auto')
        # TensorBoard callback generates at each epochs the tensorboard files
        tsboard = TensorBoard(log_dir=os.path.join(self.model_path, 'tsboard'), histogram_freq=0, batch_size=self.ann_parameters['batch_size'], write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

        # Compute callbacks list from callbacks

        callbacks_list = [checkpoint, self.custom_metrics]

        if self.tsboard:
            callbacks_list += [tsboard]
        if self.early_stop:
            callbacks_list += [earlystop]
        if self.lrdecay:
            callbacks_list += [lrdecay]

        if self.ann_parameters['weighted']:
            self.history = self.model.fit(self.X_train, self.Y_train, sample_weight=self.sample_weights_train, validation_data=(self.X_val, self.Y_val, self.sample_weights_val), epochs=self.ann_parameters['epochs'], batch_size=self.ann_parameters['batch_size'], callbacks=callbacks_list, verbose=2)
        else:
            self.history = self.model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val), epochs=self.ann_parameters['epochs'], batch_size=self.ann_parameters['batch_size'], callbacks=callbacks_list, verbose=2)

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
        plt.savefig(os.path.join(self.model_path, 'figures', 'ANN_validation_score.png'))

        plt.clf()

        # loss
        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(os.path.join(self.model_path, 'figures', 'ANN_loss.png'))

        plt.close()

        return

    def evaluate_ANN(self):

        # evaluate the model
        Y_pred = self.model.predict(self.X_val)

        if self.compute_conf_matr:
            mean_auc_roc, mean_auc_pr = compute_aucs(Y_pred, self.Y_val, self.DES_id_val, self.classnames, os.path.join(self.model_path + 'figures'), plot=True)
        else:
            mean_auc_roc, mean_auc_pr = compute_aucs(Y_pred, self.Y_val, self.DES_id_val, self.classnames)

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

        report2csv(self.model_index, os.path.join(self.script_dir, 'model_ANN'), self.catalog_filename, self.constraints, self.ann_parameters, self.classification_problem, self.training_testing_split, self.dataset_idx, self.cv_fold_nbr, self.others_flag, self.data_imputation, self.model_path, self.preprocessing_method, self.early_stopped_epoch, mean_auc_roc, mean_auc_pr, self.custom_metrics)

        # subprocess.Popen(r'explorer /select,' + model_path)
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
        self.model.load_weights(get_model_weights_path(self.model_path, weights_flag))
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
        clf.fit(self.X_train, self.Y_train)

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
        y_true, y_pred = self.Y_val, clf.predict(self.X_test)
        print(classification_report(y_true, y_pred))
        print()

        sys.stdout.close()

        return


def compute_confusion_matrix(classifier, model_index, weights_flag='final'):

        script_path = os.path.realpath(__file__)
        script_dir, _ = os.path.split(script_path)
        model_path = os.path.join(script_dir, 'model_' + classifier, str(model_index))
        if classifier == 'ANN':
            with open(os.path.join(model_path, 'ANN_architecture.json')) as f:
                ANN_architecture = json.load(f)
            with open(os.path.join(model_path,'ANN_parameters.json')) as f:
                ANN_parameters = json.load(f)
            models_inputs = csv2dict_list(os.path.join(script_dir, 'model_ANN', 'Benchmark_ANN_inputs.csv'))
            model_input = models_inputs[model_index - 1]
            model_input_formatted = format_csv_dict(model_input)
            print(model_input_formatted)

            restored_object = Classification(classifier='ANN',
                                             preprocessing_method=model_input_formatted['preprocessing_method'],
                                             catalog_filename=model_input_formatted['catalog'],
                                             classification_problem=model_input_formatted['classification_problem'],
                                             others_flag=model_input_formatted['others_flag'],
                                             constraints=model_input_formatted['constraints'],
                                             data_imputation=model_input_formatted['data_imputation'],
                                             training_testing_split=model_input_formatted['training_testing_split'],
                                             dataset_idx=1,
                                             cv_fold_nbr=1,
                                             cross_validation=False,
                                             tsboard=False,
                                             early_stop=False,
                                             lrdecay=False,
                                             compute_conf_matr=True)
            if restored_object.preprocessing_method is not None:
                restored_object.load_preprocessed_dataset()
            else:
                restored_object.load_dataset()
            restored_object.Y_train, restored_object.Y_val, restored_object.Y_test = one_hot_encode(restored_object.Y_train, restored_object.Y_val, restored_object.Y_test)
            restored_object.model_index = model_index
            restored_object.model_path = model_path
            restored_object.load_ANN(weights_flag)

            Y_pred = restored_object.model.predict(restored_object.X_val)
            mean_auc_roc, mean_auc_pr = compute_aucs(Y_pred, restored_object.Y_val, restored_object.DES_id_val, restored_object.classnames, savepath=os.path.join(restored_object.model_path,'figures'), plot=True)

        return

# Default ANN run without preprocessing

# BG_ELG_LRG_QSO_classification = Classification(others_flag='no')
# BG_ELG_LRG_QSO_classification.load_dataset()
# BG_ELG_LRG_QSO_classification.run()

# compute_confusion_matrix('ANN', 1, weights_flag='best')

# ANN run with preprocessing

# preprocessing_method = [{'method': 'SMOTE_oversampling', 'arguments':['auto',5]}]

# BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method)
# BG_ELG_LRG_QSO_classification.load_dataset()
# BG_ELG_LRG_QSO_classification.run()

# ANN run on preprocessed dataset

# preprocessing_method = [{'method': 'SMOTE_oversampling', 'arguments':['auto',5]}]

# BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=i)
# BG_ELG_LRG_QSO_classification.load_preprocessed_dataset()
# BG_ELG_LRG_QSO_classification.run()

# ANN gridsearch

BG_ELG_LRG_QSO_classification = Classification(others_flag='no')
BG_ELG_LRG_QSO_classification.load_dataset()
BG_ELG_LRG_QSO_classification.run_ANN_gridsearch()

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
#                          [{'method': 'SMOTE_oversampling', 'arguments':['auto', 5]}],
#                          [{'method': 'ADASYN_oversampling', 'arguments':['auto', 5]}],
#                          [{'method': 'RANDOM_oversampling', 'arguments':['auto']}],
#                          [{'method': 'ENN_undersampling', 'arguments':['majority', 6, 'all']}],
#                          [{'method': 'Allknn_undersampling', 'arguments':['majority', 6, 'all']}],
#                          [{'method': 'Tomek_undersampling', 'arguments':['majority']}],
#                          [{'method': 'RANDOM_undersampling', 'arguments':['majority']}],
#                          [{'method': 'CENTROID_undersampling', 'arguments':['majority']}],
#                          [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 1]}],
#                          [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 2]}],
#                          [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 3]}],
#                          [{'method': 'IHT_undersampling', 'arguments':['majority', 'adaboost', 5]}],
#                          [{'method': 'ENN_undersampling', 'arguments':['majority', 3, 'all']}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}],
#                          [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 3]}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}],
#                          [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 1]}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}]]
#
# for i in preprocessing_methods:
#     BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=i, early_stop=True, lrdecay=True)
#     BG_ELG_LRG_QSO_classification.load_dataset()

preprocessing_methods = [[{'method': 'ADASYN_oversampling', 'arguments':['auto', 5]}],
                         [{'method': 'RANDOM_oversampling', 'arguments':['auto']}],
                         [{'method': 'ENN_undersampling', 'arguments':['majority', 6, 'all']}],
                         [{'method': 'Allknn_undersampling', 'arguments':['majority', 6, 'all']}],
                         [{'method': 'Tomek_undersampling', 'arguments':['majority']}],
                         [{'method': 'RANDOM_undersampling', 'arguments':['majority']}],
                         [{'method': 'CENTROID_undersampling', 'arguments':['majority']}],
                         [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 1]}],
                         [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 2]}],
                         [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 3]}],
                         [{'method': 'IHT_undersampling', 'arguments':['majority', 'adaboost', 5]}],
                         [{'method': 'ENN_undersampling', 'arguments':['majority', 3, 'all']}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}],
                         [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 3]}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}],
                         [{'method': 'NearMiss_undersampling', 'arguments':['majority', 6, 6, 1]}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}]]

for i in preprocessing_methods:
    BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=i, early_stop=True, lrdecay=True)
    BG_ELG_LRG_QSO_classification.load_dataset()

    # def ANN_batch_generator(self):
    #
    #     # Create empty arrays to contain batch of features and labels#
    #     batch_features = np.zeros((self.ann_parameters['batch_size'], self.input_dimensions))
    #     batch_labels = np.zeros((self.ann_parameters['batch_size'],1))
    #     nbr_objects = self.Y_train.shape[0]
    #     unique, counts = np.unique(self.Y_train, return_counts=True)
    #     class_count_dict = dict(zip(unique, counts))
    #     class_count_bacth = {}
    #     isclass_Y_val = {}
    #     for label in list(class_count_dict.keys()):
    #         isclass_Y_val[label] = np.where((Y_train==label))[0]
    #         class_count_batch[label] = math.floor(self.ann_parameters['batch_size']*class_count_dict[label]/nbr_objects)
    #     iterator = 0
    #     while True:
    #         index_batch = np.zeros(self.ann_parameters['batch_size'])
    #         previous_class_count_batch = 0
    #         for label in list(class_count_dict.keys()):
    #             index_batch[previous_class_count_batch:class_count_batch[label]] = isclass_Y_val[label][iterator*class_count_batch[label]:class_count_batch[label]]
    #             previous_class_count_batch = class_count_batch[label]
    #         for i in range(batch_size):
    #             batch_features[i] = self.X_train[i, :]
    #             batch_labels[i] = self.Y_train[i]
    #         yield batch_features, batch_labels
    #     return

    # def cross_validate(layers_list, ann_parameters, dataset_path, classification_problem, train_test_split, validation_split, model_path, class_fraction):
    #
    #     directories = os.listdir (os.path.join(self.script_dir, 'model'))
    #     model_directory = str(len(directories)+1)
    #     self.model_path = os.path.join(self.script_dir, 'model', model_directory)
    #     os.makedirs(self.model_path)
    #
    #     ann_parameters_json = ann_parameters
    #     del ann_parameters_json['optimizer']
    #     with open(os.path.join(self.model_path, 'ann_parameters.json'), 'w') as fp:
    #         json.dump(ann_parameters_json, fp)
    #
    #     all_scores = []
    #     nbr_fold = int(math.floor(100/self.training_testing_split[1]))
    #
    #     for i in range(1, nbr_fold + 1, 1):
    #         self.fold_nbr = i
    #         checkpoints_name = (str(self.training_testing_split[0]) + '_'
    #                            + str(self.training_testing_split[1]) + '_'
    #                            + str(i) + '_'
    #                            + '{epoch:02d}-epo'
    #                            + '_{val_categorical_accuracy:.2f}-loss'
    #                            + '.hdf5')
    #
    #         checkpoint = ModelCheckpoint(os.path.join(self.model_path, 'checkpoints', checkpoints_name), monitor='val_'+ self.ann_parameters['metrics'][0], verbose=2, save_best_only=True, mode='max')
    #         lrate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001)
    #         earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    #         tsboard = TensorBoard(log_dir=os.path.join(self.model_path, 'tsboard'), histogram_freq=0, batch_size=self.ann_parameters['batch_size'], write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
    #         callbacks_list = [checkpoint]
    #
    #         if self.tsboard:
    #             callbacks_list += [tsboard]
    #         if self.early_stop:
    #             callbacks_list += [early_stop]
    #         if self.lrdecay:
    #             callbacks_list += [lrdecay]
    #
    #         X_train, Y_train, X_val, Y_val, sample_weights_val, X_test, Y_test, sample_weights_train = self.load_dataset()
    #         self.Y_train, self.Y_val, self.Y_test = one_hot_encode(self.Y_train, self.Y_val, self.Y_test)
    #
    #         if ann_parameters['weighted']:
    #             # Fit the model
    #             self.history = model.fit(X_train, Y_train, sample_weight=sample_weights_train, validation_data=(X_val, Y_val, sample_weights_val), epochs=ann_parameters['epochs'], batch_size=ann_parameters['batch_size'], callbacks=callbacks_list, verbose=1)
    #         else:
    #             self.history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=ann_parameters['epochs'], batch_size=ann_parameters['batch_size'], callbacks=callbacks_list, verbose=1)
    #
    #         # evaluate the model
    #         scores = model.evaluate(X_test, Y_test)
    #         all_scores.append(scores[1]*100)
    #
    #     mean_score = sum(all_scores)/len(all_scores)
    #
    #     return
