import tensorflow as tf

# XLA-compatible impl of tf.histogram_fixed_width()
def _compute_histogram(values, value_range, nbins=256):
    min_val, max_val = value_range
    
    # Prevent division by zero
    bin_width = tf.maximum(
        (max_val - min_val) / tf.cast(nbins, values.dtype), 
        tf.keras.backend.epsilon()
    )
    
    # Compute bin indices
    bins = tf.floor((values - min_val) / bin_width)
    bins = tf.cast(tf.clip_by_value(bins, 0, nbins - 1), tf.int32)
    
    # One-hot encode and sum
    encoded = tf.one_hot(bins, depth=nbins, dtype=tf.int32)
    histogram = tf.reduce_sum(encoded, axis=0)
    
    return histogram

class NormalizedHistogram(tf.keras.layers.Layer):
    def __init__(self, nbins=256, **kwargs):
        super().__init__(**kwargs)
        self.nbins = nbins

    def call(self, inputs):
        
        def op_per_image(image):
            histograms = []
            for c in range(3):
                channel = image[..., c]
                channel = tf.reshape(channel, [-1])

                hist = _compute_histogram(channel, value_range=[0.0, 1.0], nbins=self.nbins)
                hist = tf.cast(hist, tf.float32)
                hist = hist / tf.reduce_sum(hist)

                histograms.append(hist)

            histograms = tf.stack(histograms, axis=-1)
            return histograms
        
        histograms = tf.map_fn(op_per_image, inputs)
        return histograms

    def compute_output_shape(self, input_shape):
        # input: (None, h, w, 3)
        # output: (None, 256, 3)
        return (None, self.nbins, 3)
    
    def get_config(self):
        config = super().get_config()
        config.update({"nbins": self.nbins, 'trainable': False})
        return config
