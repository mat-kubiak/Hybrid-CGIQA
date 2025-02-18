import tensorflow as tf
from tensorflow.keras.layers import Layer

class RGBToHSV(Layer):
    """
    Simple layer that computes RGB input image into HSV space.
    
    In case you want to use histograms, we found out that they are way more effective in the HSV space.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, 0.0, 1.0)
        return tf.image.rgb_to_hsv(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({'trainable': False})
        return config
