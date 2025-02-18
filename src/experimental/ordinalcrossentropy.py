import tensorflow as tf
from keras.losses import categorical_crossentropy

class OrdinalCrossentropy(tf.keras.losses.Loss):
    """
    Ordinal crossentropy; can be used as a loss function for ordinal classification.
    
    In case you want to train the network as a classification of score ranges, rather than regression,
    we recommend using this loss, because it adds context of which classes are similar and which are further apart.
    This is not the case for the standard crossentropy loss.
    """
    def call(self, y_true, y_pred):
        true_cat = tf.argmax(y_true, axis=1)
        pred_cat = tf.argmax(y_pred, axis=1)

        cats = y_pred.shape[1]

        weights = tf.abs(true_cat - pred_cat)/(cats - 1)
        weights_casted = tf.cast(weights, dtype='float32')

        return (1.0 + weights_casted) * categorical_crossentropy(y_true, y_pred)
