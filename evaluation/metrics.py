import tensorflow as tf


def metrics_prepare(labels, predictions, prediction_type='regression', five2two=False):
    """preprocessing of predictions"""

    labels = tf.cast(labels, dtype=tf.int32)

    # change the original predictions into the standard 5-class predictions
    if prediction_type == 'regression':  # from float to int
        predictions = tf.cast(tf.clip_by_value(predictions + 0.5, clip_value_min=0, clip_value_max=4), dtype=tf.int32)
    elif prediction_type == 'classification':  # from probability to int
        predictions = tf.reshape(tf.math.argmax(predictions, axis=1), shape=(-1, 1))
        predictions = tf.cast(predictions, dtype=tf.int32)

    # change the standard 5-class predictions into the standard binary predictions
    if five2two:
        labels = tf.cast(labels >= 2, dtype=tf.int32)
        predictions = tf.cast(predictions >= 2, dtype=tf.int32)

    return labels, predictions


class BinaryAccuracy(tf.keras.metrics.Metric):
    """metric: binary accuracy"""

    def __init__(self, model_type, name='binary_accuracy', **kwargs):
        super(BinaryAccuracy, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.model_type = model_type
        self.binary_accuracy = self.add_weight(name='binary_accuracy', initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # update parameters
        if self.model_type == 'regression':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='regression', five2two=True)
        elif self.model_type == 'binary_classification':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='classification', five2two=False)
        elif self.model_type == 'multi_classification':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='classification', five2two=True)

        self.binary_accuracy = tf.math.reduce_mean(tf.cast(tf.math.equal(self.labels, self.predictions), dtype=tf.float32))

    def result(self):
        # return parameters
        return self.binary_accuracy


class MultiAccuracy(tf.keras.metrics.Metric):
    """metric: multi-accuracy"""

    def __init__(self, model_type, name='multi_accuracy', **kwargs):
        super(MultiAccuracy, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.model_type = model_type
        self.multi_accuracy = self.add_weight(name='multi_accuracy', initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # update parameters
        if self.model_type == 'regression':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='regression', five2two=False)
        elif self.model_type == 'multi_classification':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='classification', five2two=False)

        self.multi_accuracy = tf.math.reduce_mean(tf.cast(tf.math.equal(self.labels, self.predictions), dtype=tf.float32))

    def result(self):
        # return parameters
        return self.multi_accuracy


class BinaryConfusionMatrix(tf.keras.metrics.Metric):
    """metric: binary confusion matrix"""

    def __init__(self, model_type, name='binary_confusion_matrix', **kwargs):
        super(BinaryConfusionMatrix, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.model_type = model_type
        self.binary_confusion_matrix = self.add_weight(name='binary_confusion_matrix', shape=(2, 2), initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # update parameters
        if self.model_type == 'regression':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='regression', five2two=True)
        elif self.model_type == 'binary_classification':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='classification', five2two=False)
        elif self.model_type == 'multi_classification':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='classification', five2two=True)

        self.binary_confusion_matrix = tf.math.confusion_matrix(tf.squeeze(self.labels), tf.squeeze(self.predictions), num_classes=2, dtype=tf.int32)

    def result(self):
        # return parameters
        return self.binary_confusion_matrix


class MultiConfusionMatrix(tf.keras.metrics.Metric):
    """metric: multi-confusion-matrix"""

    def __init__(self, model_type, name='multi_confusion_matrix', **kwargs):
        super(MultiConfusionMatrix, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.model_type = model_type
        self.multi_confusion_matrix = self.add_weight(name='multi_confusion_matrix', shape=(5, 5), initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # update parameters
        if self.model_type == 'regression':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='regression', five2two=False)
        elif self.model_type == 'multi_classification':
            self.labels, self.predictions = metrics_prepare(labels, predictions, prediction_type='classification', five2two=False)

        self.multi_confusion_matrix = tf.math.confusion_matrix(tf.squeeze(self.labels), tf.squeeze(self.predictions), num_classes=5, dtype=tf.int32)

    def result(self):
        # return parameters
        return self.multi_confusion_matrix
