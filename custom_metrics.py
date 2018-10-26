# This module defines a special keras callback that computes metrics that are not
# available in keras defautl Metrics. Being a callback, those methods are called
# between each epochs which can slow down the training of the model
# For mor info about keras callbacks and their attributes : https://keras.io/callbacks/

import keras
import numpy as np
from sklearn.metrics import classification_report
from utils import *

class Metrics(keras.callbacks.Callback):

    # Method run at beginning of train

    def on_train_begin(self, logs={}):
        self.macro_f1s = []
        self.macro_precision = []
        self.macro_recall = []
        self.epochs = []

    # Method run at the end of each epoch

    def on_epoch_end(self, epoch, logs={}):
        Y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=-1)
        Y_val = np.argmax(self.validation_data[1], axis=-1)

        tmp_report = classification_report(Y_val, Y_pred)
        reportdict = report2dict(tmp_report)
        self.epochs.append(epoch + 1)
        self.macro_f1s.append(float(reportdict[' macro avg']['f1-score']))
        self.macro_precision.append(float(reportdict[' macro avg']['precision']))
        self.macro_recall.append(float(reportdict[' macro avg']['recall']))
        print('Epoch : ', self.epochs[-1], ' Macro precision : ', self.macro_precision[-1], ' Macro recall : ', self.macro_recall[-1], ' Macro f1-Score : ', self.macro_f1s[-1])

        return
