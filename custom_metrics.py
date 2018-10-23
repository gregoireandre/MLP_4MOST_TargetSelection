import keras
import numpy as np
from sklearn.metrics import classification_report
from utils import *



class Metrics(keras.callbacks.Callback):


    def on_train_begin(self, logs={}):
        self.macro_f1s = []
        self.macro_precision = []
        self.macro_recall = []
        self.epochs = []

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
