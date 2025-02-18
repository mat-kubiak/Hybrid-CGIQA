import tensorflow as tf

def _compute_histogram(values, value_range, nbins=256):
    """XLA-compatible impl of tf.histogram_fixed_width()"""

    min_val, max_val = value_range
    
    bin_width = tf.maximum(
        (max_val - min_val) / tf.cast(nbins, values.dtype), 
        tf.keras.backend.epsilon()
    )
    
    bins = tf.floor((values - min_val) / bin_width)
    bins = tf.cast(tf.clip_by_value(bins, 0, nbins - 1), tf.int32)
    
    encoded = tf.one_hot(bins, depth=nbins, dtype=tf.int32)
    histogram = tf.reduce_sum(encoded, axis=0)
    
    return histogram

class NormalizedHistogram(tf.keras.layers.Layer):
    """Custom histogram layer that outputs normalized info

    This layer can work with images of any dimensions and any number of channels,
    provided that it is in the 'channels-last' format.
    """
    def __init__(self, nbins=256, **kwargs):
        super().__init__(**kwargs)
        self.nbins = nbins

    def call(self, inputs):

        def op_per_channel(channel):
            flattened = tf.reshape(channel, [-1])

            hist = _compute_histogram(flattened, value_range=[0.0, 1.0], nbins=self.nbins + 1)
            hist = tf.cast(hist, tf.float32)
            hist = hist[1:]  # Remove the first bin to remove impact of zero-padding

            # prevent div by 0 when channel is empty
            denom = tf.maximum(tf.reduce_sum(hist), tf.keras.backend.epsilon())
            return hist / denom

        def op_per_image(image):
            channels_first = tf.transpose(image, perm=[2, 0, 1])
            histograms = tf.map_fn(op_per_channel, channels_first, dtype=tf.float32)
            return tf.transpose(histograms, perm=[1, 0])

        histograms = tf.map_fn(op_per_image, inputs, dtype=tf.float32)
        return histograms

    def compute_output_shape(self, input_shape):
        samples, height, width, channels = input_shape
        return (samples, self.nbins, channels)

    def get_config(self):
        config = super().get_config()
        config.update({"nbins": self.nbins, 'trainable': False})
        return config
