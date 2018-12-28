# This module defines a special keras callback that computes metrics that are not
# available in keras defautl Metrics. Being a callback, those methods are called
# between each epochs which can slow down the training of the model
# For mor info about keras callbacks and their attributes : https://keras.io/callbacks/

import keras
import numpy as np
from utils_regression import *

class Metrics(keras.callbacks.Callback):

    # Method run at beginning of train

    def on_train_begin(self, logs={}):
        self.epochs = []
        self.variance = []
        self.bias = []

    # Method run at the end of each epoch

    def on_epoch_end(self, epoch, logs={}):
        Y_pred = self.model.predict(self.validation_data[0])
        Y_val = self.validation_data[1]

        Y_pred_squared = [i**2 for i in Y_pred]
        variance = np.mean(Y_pred_squared) - np.mean(Y_pred)**2
        error = abs(Y_pred - Y_val)
        bias = np.mean(error)
        self.epochs.append(epoch)
        self.bias.append(bias)
        self.variance.append(variance)
        print('Epoch : ', self.epochs[-1], ' Bias : ', self.bias[-1], ' Variance : ', self.variance[-1])
        return
