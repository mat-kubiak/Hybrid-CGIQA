import tensorflow as tf
from tensorflow.keras import layers

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(filters=1,
                           kernel_size=self.kernel_size,
                           strides=1,
                           padding="same",
                           activation="sigmoid")
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        
        attention_map = self.conv(concat)

        output = layers.Multiply()([inputs, attention_map])
        return output

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config
