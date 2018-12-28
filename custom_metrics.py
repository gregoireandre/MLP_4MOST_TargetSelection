# This module defines a special keras callback that computes metrics that are not
# available in keras defautl Metrics. Being a callback, those methods are called
# between each epochs which can slow down the training of the model
# For mor info about keras callbacks and their attributes : https://keras.io/callbacks/

import keras
import numpy as np
from sklearn.metrics import classification_report
from utils_classification import *

class Metrics(keras.callbacks.Callback):

    # Method run at beginning of train

    def on_train_begin(self, logs={}):
        self.macro_f1s = []
        self.macro_precision = []
        self.macro_recall = []
        self.epochs = []
        self.no_others_precision = []
        self.no_others_recall = []
        self.no_others_f1 = []
        self.elgs = {'precision': [], 'recall': [], 'f1': []}
        self.lrgs = {'precision': [], 'recall': [], 'f1': []}
        self.bgs = {'precision': [], 'recall': [], 'f1': []}
        self.qsos = {'precision': [], 'recall': [], 'f1': []}
        self.others = {'precision': [], 'recall': [], 'f1': []}

    # Method run at the end of each epoch

    def on_epoch_end(self, epoch, logs={}):
        Y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=-1)
        Y_val = np.argmax(self.validation_data[1], axis=-1)

        reportdict = classification_report(Y_val, Y_pred, output_dict=True)
        print(list(reportdict.keys()))
        self.epochs.append(epoch + 1)
        self.macro_f1s.append(float(reportdict['macro avg']['f1-score']))
        self.macro_precision.append(float(reportdict['macro avg']['precision']))
        self.macro_recall.append(float(reportdict['macro avg']['recall']))
        self.others['precision'].append(float(reportdict['0']['precision']))
        self.others['recall'].append(float(reportdict['0']['recall']))
        self.others['f1'].append(float(reportdict['0']['f1-score']))
        self.bgs['precision'].append(float(reportdict['1']['precision']))
        self.bgs['recall'].append(float(reportdict['1']['recall']))
        self.bgs['f1'].append(float(reportdict['1']['f1-score']))
        self.lrgs['precision'].append(float(reportdict['2']['precision']))
        self.lrgs['recall'].append(float(reportdict['2']['recall']))
        self.lrgs['f1'].append(float(reportdict['2']['f1-score']))
        self.elgs['precision'].append(float(reportdict['3']['precision']))
        self.elgs['recall'].append(float(reportdict['3']['recall']))
        self.elgs['f1'].append(float(reportdict['3']['f1-score']))
        self.qsos['precision'].append(float(reportdict['4']['precision']))
        self.qsos['recall'].append(float(reportdict['4']['recall']))
        self.qsos['f1'].append(float(reportdict['4']['f1-score']))
        self.no_others_precision.append((self.elgs['precision'][-1] + self.lrgs['precision'][-1] + self.bgs['precision'][-1] + self.qsos['precision'][-1])/4)
        self.no_others_recall.append((self.elgs['recall'][-1] + self.lrgs['recall'][-1] + self.bgs['recall'][-1] + self.qsos['recall'][-1])/4)
        self.no_others_f1.append((self.elgs['f1'][-1] + self.lrgs['f1'][-1] + self.bgs['f1'][-1] + self.qsos['f1'][-1])/4)
        print('Epoch : ', self.epochs[-1], ' Macro precision : ', self.macro_precision[-1], ' Macro recall : ', self.macro_recall[-1], ' Macro f1-Score : ', self.macro_f1s[-1])
        print('Epoch : ', self.epochs[-1], ' NO Macro precision : ', self.no_others_precision[-1], ' NO Macro recall : ', self.no_others_recall[-1], ' NO Macro f1-Score : ', self.no_others_f1[-1])
        return
