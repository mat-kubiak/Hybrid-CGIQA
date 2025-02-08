import tensorflow as tf

class AdaptiveAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, grid_size=2, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        target_h = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) / self.grid_size), tf.int32) * self.grid_size
        target_w = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) / self.grid_size), tf.int32) * self.grid_size

        pad_h = target_h - height
        pad_w = target_w - width
        
        paddings = [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
        padded_inputs = tf.pad(inputs, paddings, "REFLECT")
        
        h_splits = target_h // self.grid_size
        w_splits = target_w // self.grid_size

        shape1 = [batch_size, self.grid_size, h_splits, target_w, channels]
        splits = tf.reshape(padded_inputs, shape1)

        shape2 = [batch_size, self.grid_size, h_splits, self.grid_size, w_splits, channels]
        splits = tf.reshape(splits, shape2)

        pooled = tf.reduce_mean(splits, axis=2)
        pooled = tf.reduce_mean(pooled, axis=3)

        return pooled

    def compute_output_shape(self, input_shape):
        samples, _, _, channels = input_shape
        return (samples, self.grid_size, self.grid_size, channels)
    
    def get_config(self):
        config = super().get_config()
        config.update({"grid_size": self.grid_size})
        return config
