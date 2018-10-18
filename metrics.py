
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

# class Metrics(keras.callbacks.Callback):
#
#     def on_train_begin(self, logs={}):
#         self.confusion = []
#         self.precision = []
#         self.recall = []
#         self.f1s = []
#         self.kappa = []
#         self.auc = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         predicted = np.argmax(self.model.predict(self.validation_data[0]), axis=-1)
#         ground_truth = self.validation_data[1]
#
#         self.auc.append(sklm.roc_auc_score(targ, score))
#         self.confusion.append(sklm.confusion_matrix(targ, predict))
#         self.precision.append(sklm.precision_score(targ, predict))
#         self.recall.append(sklm.recall_score(targ, predict))
#         self.f1s.append(sklm.f1_score(targ, predict))
#         self.kappa.append(sklm.cohen_kappa_score(targ, predict))
#
#         return
# def compute_batch(X, Y, batch_size):
#
#     unique, counts = np.unique(Y, return_counts=True)
#     nbr_objects = Y.shape[0]
#     class_count_dict = dict(zip(unique, counts))
#     class_count_dict[]
#     all_labels = list(class_count_dict.keys())
#     sample_weights = []
#
#     for i in Y:
#         for j in all_labels:
#             if i == j:
#                 # if i == 0.0:
#                 #     sample_weights.append(class_weights[j])
#                 # else:
#                 #     sample_weights.append(class_weights[j])
#                 sample_weights.append(math.sqrt(class_weights[j]))

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())
    # return precision
    # y_true = K.print_tensor(y_true, message='')
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # class_id_true = K.print_tensor(class_id_true, message='')
    # Replace class_id_preds with class_id_true for recall here
    mask = K.cast(K.not_equal(class_id_preds, 0), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(mask), 1)

    return class_acc
