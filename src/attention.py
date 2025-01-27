import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Activation, Multiply, Add, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape, Concatenate

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv2D(filters=1,
                           kernel_size=self.kernel_size,
                           strides=1,
                           padding="same",
                           activation="sigmoid")
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass of the layer.

        Args:
            inputs (tf.Tensor): Input feature map of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output feature map after applying spatial attention.
        """
        # Compute average pooling and max pooling along the channel axis
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)  # Shape: (batch, H, W, 1)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)  # Shape: (batch, H, W, 1)
        
        # Concatenate average and max pooling results along the channel axis
        concat = Concatenate(axis=-1)([avg_pool, max_pool])  # Shape: (batch, H, W, 2)
        
        # Generate the spatial attention map using a convolution layer
        attention_map = self.conv(concat)  # Shape: (batch, H, W, 1)

        # Multiply the attention map with the input feature map
        output = Multiply()([inputs, attention_map])
        return output

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config
