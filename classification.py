import __init__
import preprocessing
from utils import *
from custom_metrics import *

import os
import sys
import math
import json
import keras
import random
import datetime
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits


class Classification():

    def __init__(self, classifier='ANN', preprocessing_method=None, random_seed=7, catalog_filename='4MOST.CatForGregoire.11Oct2018.zphot', classification_problem='BG_ELG_LRG_QSO_classification', constraints='no', others_flag='all', training_testing_split=[80, 20], dataset_idx=1, cv_fold_nbr=1, cross_validation=False, tsboard=False, early_stop=False, lrdecay=True):

        self.classifier = classifier
        self.preprocessing_method = preprocessing_method
        self.random_seed = random_seed
        self.catalog_filename = catalog_filename
        self.classification_problem = classification_problem
        self.constraints= constraints
        self.others_flag = others_flag
        self.training_testing_split = training_testing_split
        self.dataset_idx = dataset_idx
        self.cv_fold_nbr = cv_fold_nbr
        self.cross_validation = cross_validation
        self.tsboard = tsboard
        self.early_stop = early_stop
        self.lrdecay = lrdecay

        self.seed = 7
        np.random.seed(self.seed)
        self.script_path = os.path.realpath(__file__)
        self.script_dir, _ = os.path.split(self.script_path)
        self.constraints = constraints_to_str(self.constraints)
        self.classnames = compute_classnames(self.classification_problem, self.others_flag)

        print('Initialized, will load dataset')

        self.dataset_path = os.path.join(self.script_dir, 'datasets', self.catalog_filename, self.others_flag + '-others_' + self.constraints + '-constraints')

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
                    self.X_train, self.Y_train = getattr(preprocessing, i['method'])(self.X_train, self.Y_train, self.seed, i['arguments'][0])
                elif len(i['arguments']) == 2:
                    self.X_train, self.Y_train = getattr(preprocessing, i['method'])(self.X_train, self.Y_train, self.seed, i['arguments'][0], i['arguments'][1])
                elif len(i['arguments']) == 3:
                    self.X_train, self.Y_train = getattr(preprocessing, i['method'])(self.X_train, self.Y_train, self.seed, i['arguments'][0], i['arguments'][1], i['arguments'][2])
                elif len(i['arguments']) == 4:
                    self.X_train, self.Y_train = getattr(preprocessing, i['method'])(self.X_train, self.Y_train, self.seed, i['arguments'][0], i['arguments'][1], i['arguments'][2], i['arguments'][3])
                preprocessing.save_preprocessed_dataset(self.script_dir, self.catalog_filename, self.others_flag, self.constraints, self.classification_problem, self.training_testing_split, self.dataset_idx, self.cv_fold_nbr, i['method'], i['arguments'], self.X_train, self.Y_train)
        self.sample_weights_train = compute_weights(self.Y_train)

        self.input_dimensions = self.X_train.shape[1]
        print(self.input_dimensions)
        self.nbr_classes = np.unique(self.Y_test).shape[0]

        return

    def init_ANN(self):

        if self.ann_parameters['SNN']:
            model = Sequential()
            for idx,i in enumerate(self.layers_list):
                if 'input_dim' in i.keys():
                    model.add(Dense(i['size'], input_dim=i['input_dim'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['normalize']:
                        model.add(BatchNormalization())
                    model.add(AlphaDropout(self.ann_parameters['dropout']))
                elif(idx == len(self.layers_list) - 1):
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                else:
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['normalize']:
                        model.add(BatchNormalization())
                    model.add(AlphaDropout(self.ann_parameters['dropout']))
            # Compile model
            model.compile(loss=self.ann_parameters['loss_function'], optimizer=self.ann_parameters['optimizer'], metrics=self.ann_parameters['metrics'])
        else:
            model = Sequential()
            for idx,i in enumerate(self.layers_list):
                if 'input_dim' in i.keys():
                    model.add(Dense(i['size'], input_dim=i['input_dim'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['normalize']:
                        model.add(BatchNormalization())
                    model.add(Dropout(self.ann_parameters['dropout'], None, self.seed))
                elif(idx == len(self.layers_list) - 1):
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                else:
                    model.add(Dense(i['size'], activation=i['activation'], use_bias=False, kernel_initializer=self.ann_parameters['kernel_initializer'], bias_initializer=self.ann_parameters['bias_initializer'], kernel_regularizer=self.ann_parameters['kernel_regularizer'], bias_regularizer=self.ann_parameters['bias_regularizer'], kernel_constraint=self.ann_parameters['kernel_constraint'], bias_constraint=self.ann_parameters['bias_constraint']))
                    if self.ann_parameters['normalize']:
                        model.add(BatchNormalization())
                    model.add(Dropout(self.ann_parameters['dropout'], None, self.seed))
            # Compile model

        self.model = model

        return

    def train_ANN(self):

        if not os.path.exists(os.path.join(self.script_dir, 'model_ANN')):
            os.makedirs(os.path.join(self.script_dir, 'model_ANN'))

        nbr_directories = len(next(os.walk(os.path.join(self.script_dir, 'model_ANN')))[1])
        model_directory = str(nbr_directories+1)
        self.model_path = os.path.join(self.script_dir, 'model_ANN', model_directory)

        os.makedirs(self.model_path)

        if not os.path.exists(os.path.join(self.model_path, 'tsboard')):
            os.makedirs(os.path.join(self.model_path, 'tsboard'))

        if not os.path.exists(os.path.join(self.model_path, 'figures')):
            os.makedirs(os.path.join(self.model_path, 'figures'))

        if not os.path.exists(os.path.join(self.model_path, 'checkpoints')):
            os.makedirs(os.path.join(self.model_path, 'checkpoints'))

        self.Y_train, self.Y_val, self.Y_test = one_hot_encode(self.Y_train, self.Y_val, self.Y_test)
        custom_metrics = Metrics()

        ann_parameters_json = self.ann_parameters.copy()
        del ann_parameters_json['optimizer']
        del ann_parameters_json['kernel_regularizer']
        ann_parameters_json['metrics'] = ann_parameters_json['metrics'][0]
        with open(os.path.join(self.model_path, 'ann_parameters.json'), 'w') as fp:
            json.dump(ann_parameters_json, fp)

        self.model.compile(loss=self.ann_parameters['loss_function'], optimizer=self.ann_parameters['optimizer'], metrics=[])

        # serialize model architecture to JSON
        model_json = self.model.to_json()
        with open(os.path.join(self.model_path, str(self.training_testing_split[0]) + '_' + str(self.training_testing_split[1]) + '_' + str(self.dataset_idx) + '_' + str(self.cv_fold_nbr) + '.json'), "w") as json_file:
            json_file.write(model_json)

        checkpoints_name = (str(self.training_testing_split[0]) + '_'
                           + str(self.training_testing_split[1]) + '_'
                           + str(self.dataset_idx) + '_'
                           + str(self.cv_fold_nbr) + '_'
                           + '{epoch:03d}-epo'
                           + '.hdf5')

        checkpoint = ModelCheckpoint(os.path.join(self.model_path, 'checkpoints', checkpoints_name), verbose=2, save_best_only=False)
        lrdecay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
        tsboard = TensorBoard(log_dir=os.path.join(self.model_path, 'tsboard'), histogram_freq=0, batch_size=self.ann_parameters['batch_size'], write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
        callbacks_list = [checkpoint, custom_metrics]

        if self.tsboard:
            callbacks_list += [tsboard]
        if self.early_stop:
            callbacks_list += [early_stop]
        if self.lrdecay:
            callbacks_list += [lrdecay]

        if self.ann_parameters['weighted']:
            self.history = self.model.fit(self.X_train, self.Y_train, sample_weight=self.sample_weights_train, validation_data=(self.X_val, self.Y_val, self.sample_weights_val), epochs=self.ann_parameters['epochs'], batch_size=self.ann_parameters['batch_size'], callbacks=callbacks_list, verbose=2)
        else:
            self.history = self.model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val), epochs=self.ann_parameters['epochs'], batch_size=self.ann_parameters['batch_size'], callbacks=callbacks_list, verbose=2)

        self.custom_metrics = custom_metrics

        last_f1_score = self.custom_metrics.macro_f1s[-1]
        best_f1_score = max(self.custom_metrics.macro_f1s)
        best_f1_epoch = self.custom_metrics.epochs[self.custom_metrics.macro_f1s.index(best_f1_score)]
        print(best_f1_epoch)

        directories = os.listdir(os.path.join(self.model_path, 'checkpoints'))
        for idx, i in enumerate(directories):
            if '{:03}'.format(best_f1_epoch) in i:
                print(i)
                print(best_f1_score)
                new_filename = i.split('.')[0] + '_' + str(best_f1_score) + '-macro-f1.hdf5'
                os.rename(os.path.join(self.model_path, 'checkpoints', i), os.path.join(self.model_path, 'checkpoints', new_filename))
            else:
                os.remove(os.path.join(self.model_path, 'checkpoints', i))

        self.early_stopped_epoch = earlystop.stopped_epoch

        if self.early_stopped_epoch > 0:
            # serialize weights to HDF5
            self.model.save_weights(os.path.join(self.model_path, str(self.training_testing_split[0]) + '_' + str(self.training_testing_split[1]) + '_' + str(self.dataset_idx) + '_' + str(self.cv_fold_nbr) + '_' + str(self.early_stopped_epoch) + '-epo_' + str(last_f1_score) + '-macro-f1.hdf5'))
        else:
            self.model.save_weights(os.path.join(self.model_path, str(self.training_testing_split[0]) + '_' + str(self.training_testing_split[1]) + '_' + str(self.dataset_idx) + '_' + str(self.cv_fold_nbr) + '_' + str(self.ann_parameters['epochs']) + '-epo_' + str(last_f1_score) + '-macro-f1.hdf5'))

        print("Saved model to disk")

        return

    def evaluate_ANN(self):

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
        plt.savefig(os.path.join(self.model_path, 'figures', str(self.training_testing_split[0]) + '_' + str(self.training_testing_split[1]) + '_' + str(self.dataset_idx) + '_' + str(self.cv_fold_nbr) + '_accuracy.png'))

        plt.clf()

        # loss
        plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(os.path.join(self.model_path, 'figures', str(self.training_testing_split[0]) + '_' + str(self.training_testing_split[1]) + '_' + str(self.dataset_idx) + '_' + str(self.cv_fold_nbr) + '_loss.png'))

        # evaluate the model
        Y_pred = self.model.predict(self.X_val)

        mean_auc_roc, mean_auc_pr = compute_aucs(Y_pred, self.Y_val, self.classnames)

        if Y_pred.shape[1] > 1:
            Y_pred = np.argmax(Y_pred, axis=-1)
            Y_val = np.argmax(self.Y_val, axis=-1)
        else:
            Y_pred = np.squeeze(Y_pred)
            Y_pred = (Y_pred > 0.5).astype(int)

        report = classification_report(Y_val, Y_pred, target_names=self.classnames)
        filename_report = str(self.training_testing_split[0]) + '_' + str(self.training_testing_split[1]) + '_' + str(self.dataset_idx) + '_' + str(self.cv_fold_nbr) + '_report.txt'
        with open(os.path.join(self.model_path, filename_report), "w") as fp:
            fp.write(report)

        reportdict = report2dict(report)
        report2csv(reportdict, self.catalog_filename, self.constraints, self.ann_parameters, self.classification_problem, self.training_testing_split, self.dataset_idx, self.cv_fold_nbr, self.others_flag, self.model_path, self.preprocessing_method, self.early_stopped_epoch, mean_auc_roc, mean_auc_pr)

        # subprocess.Popen(r'explorer /select,' + model_path)
        return

    def define_ANN(self):

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

        set_random_seed(self.seed)

        self.ann_parameters =  {'loss_function': 'categorical_crossentropy',
                                'learning_rate': 0.0005,
                                'optimizer': Adam(lr=0.0005),
                                'optimizer_str': 'adam',
                                'batch_size': 2048,
                                'epochs': 2,
                                'metrics': ['categorical_accuracy'],
                                'nbr_layers': 2,
                                'nbr_neurons' : 64,
                                'activation' : 'relu',
                                'output_activation': 'softmax',
                                'dropout': 0.0,
                                'kernel_initializer': 'lecun_normal',
                                'bias_initializer' : 'zeros',
                                'kernel_regularizer': None,
                                'kernel_regularizer_str': None,
                                'regularization_strength': 0.0,
                                'bias_regularizer': None,
                                'activity_regularizer': None,
                                'kernel_constraint': None,
                                'bias_constraint': None,
                                'SNN': False,
                                'weighted': False,
                                'normalize': False}

        self.layers_list = []
        self.layers_list.append({'size': self.ann_parameters['nbr_neurons'], 'input_dim': self.input_dimensions, 'activation': self.ann_parameters['activation']})
        for i in range(1,self.ann_parameters['nbr_layers'] - 1):
            self.layers_list.append({'size': self.ann_parameters['nbr_neurons'], 'activation' : self.ann_parameters['activation']})
        self.layers_list.append({'size': self.nbr_classes, 'activation': self.ann_parameters['output_activation']})

        return

    def init_RF(self):

        rf = RandomForestClassifier(random_state=self.seed)
        self.model = rf
        return

    def CV_gridsearch(self, params_dist, metric):

        sys.stdout = open(os.path.join(self.script_dir, 'model_' + self.classifier, 'CV_gridsearch_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +'.txt' , 'w'))

        print("# Tuning hyper-parameters for %s" % metric)
        print()

        clf = GridSearchCV(self.model, params_dist, n_jobs=-1, cv=10,
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


# preprocessing_method = [{'method': 'SMOTE_oversampling', 'arguments':['auto',5]}]                  # arguments are sampling_strategy, k_neighbors=5
# preprocessing_method = [{'method': 'ADASYN_oversampling', 'arguments':['auto',5]}]                 # arguments are sampling_strategy, n_neighbors=5
# preprocessing_method = [{'method': 'RANDOM_oversampling', 'arguments':['auto']}]                   # arguments are sampling_strategy
# preprocessing_method = [{'method': 'ENN_undersampling', 'arguments':['auto', 3, 'all']}]           # arguments are sampling_strategy, n_neighbors=3, kind_sel='all'
# preprocessing_method = [{'method': 'Allknn_undersampling', 'arguments':['auto', 3, 'all']}]        # arguments are sampling_strategy, n_neighbors=3, kind_sel='all'
# preprocessing_method = [{'method': 'Tomek_undersampling', 'arguments':['auto']}]                   # arguments are sampling_strategy
# preprocessing_method = [{'method': 'RANDOM_undersampling', 'arguments':['auto']}]                  # arguments are sampling_strategy
# preprocessing_method = [{'method': 'CENTROID_undersampling', 'arguments':['auto']}]                # arguments are sampling_strategy
# preprocessing_method = [{'method': 'NearMiss_undersampling', 'arguments':['auto', 3, 3, 1]}]       # arguments are sampling_strategy, n_neighbors=3, n_neighbors_ver3=3, version=1
# preprocessing_method = [{'method': 'IHT_undersampling', 'arguments':['auto', 'adaboost', 5]}]      # arguments are sampling_strategy, sampling_strategy, estimator='adaboost', cv=5
# preprocessing_method = [{'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}]                # arguments are sampling_strategy, k_neighbors_smote=5, n_neighbors_enn=3, kind_sel='all'
# preprocessing_method = [{'method': 'SMOTE_Tomek', 'arguments':['auto']}]                           # sampling_strategy, k_neighbors_smote=5
# preprocessing_method = [{'method': 'ENN_undersampling', 'arguments':['auto', 5, 3, 'all']}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}]

# Default ANN run without preprocessing

# BG_ELG_LRG_QSO_classification = Classification()
# BG_ELG_LRG_QSO_classification.load_dataset()
# BG_ELG_LRG_QSO_classification.run()

# ANN run with preprocessing
# preprocessing_method = [{'method': 'SMOTE_oversampling', 'arguments':['auto',5]}]                  # arguments are sampling_strategy, k_neighbors=5
# preprocessing_method = [{'method': 'ENN_undersampling', 'arguments':['auto', 3, 'all']}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}]
# BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=preprocessing_method)
# BG_ELG_LRG_QSO_classification.load_dataset()
# BG_ELG_LRG_QSO_classification.run()

#RF gridsearch

# BG_ELG_LRG_QSO_classification = Classification(classifier='RF')
# BG_ELG_LRG_QSO_classification.load_dataset()
# BG_ELG_LRG_QSO_classification.init_RF()
# param_dist = {'n_estimators' : [10, 20, 30]                             #Number of trees, normally the higher, the better
#               'max_depth': [2, 3, 4],                                   #Depth of forest should not be too high for noisy data
#               'bootstrap': [True, False],                               #Use bottstrap
#               'max_features': ['auto', 'log2', 8],                      #Number of features to test at each node
#               'criterion': ['gini', 'entropy']}                         #Metric used for evaluation
# cv_metric  ='f1-score'
# BG_ELG_LRG_QSO_classification.CV_gridsearch(params_dist, cv_metric)

#Benchmark processing methods

preprocessing_methods = [[{'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}],
                         [{'method': 'SMOTE_Tomek', 'arguments':['auto']}],
                         [{'method': 'SMOTE_oversampling', 'arguments':['auto', 5]}],
                         [{'method': 'ADASYN_oversampling', 'arguments':['auto', 5]}],
                         [{'method': 'RANDOM_oversampling', 'arguments':['auto']}],
                         [{'method': 'ENN_undersampling', 'arguments':['auto', 3, 'all']}],
                         [{'method': 'Allknn_undersampling', 'arguments':['auto', 3, 'all']}],
                         [{'method': 'Tomek_undersampling', 'arguments':['auto']}],
                         [{'method': 'RANDOM_undersampling', 'arguments':['auto']}],
                         [{'method': 'CENTROID_undersampling', 'arguments':['auto']}],
                         [{'method': 'CENTROID_undersampling', 'arguments':['auto']}],
                         [{'method': 'NearMiss_undersampling', 'arguments':['auto', 3, 3, 1]}],
                         [{'method': 'IHT_undersampling', 'arguments':['auto', 'adaboost', 5]}],
                         [{'method': 'ENN_undersampling', 'arguments':['auto', 3, 'all']}, {'method': 'SMOTE_ENN', 'arguments':['auto', 5, 3, 'all']}]]

for i in preprocessing_methods:
    BG_ELG_LRG_QSO_classification = Classification(preprocessing_method=i)
    BG_ELG_LRG_QSO_classification.load_dataset()
    # BG_ELG_LRG_QSO_classification.run()

# BG_ELG_LRG_QSO_classification_noothers = Classification(class_fraction=[0.0, 1.0, 1.0, 1.0, 1.0])

# constraints = {'18J19.5': [['18', '<', 'VHS_j'], #LRG Cut
#                          ['19.5', '>', 'VHS_j']]
#               }
# constraints = {'16J18': [['16', '<', 'VHS_j'], #BG Cut
#                          ['18', '>', 'VHS_j']]
#               }

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
