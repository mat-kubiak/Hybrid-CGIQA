import tensorflow as tf

class ThirdsAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_gaussian_2d(self, size=(100, 100), mean=(0.5, 0.5), sigma=(0.15, 0.15)):
        y = tf.linspace(0.0, 1.0, size[0])
        x = tf.linspace(0.0, 1.0, size[1])
        
        xx, yy = tf.meshgrid(x, y)
        
        gaussian = tf.exp(-(
            ((xx - mean[1]) ** 2) / (2 * sigma[1] ** 2) + 
            ((yy - mean[0]) ** 2) / (2 * sigma[0] ** 2)
        ))
        
        return gaussian

    def call(self, inputs):
        height, width = inputs.shape[1:3]
        dims = (height, width)

        gaussian = self.create_gaussian_2d(size=dims, mean=(0.5, 0.333), sigma=(0.17, 0.07))
        gaussian += self.create_gaussian_2d(size=dims, mean=(0.5, 0.666), sigma=(0.17, 0.07))
        gaussian = gaussian / tf.reduce_max(gaussian)
        
        gaussian = tf.expand_dims(gaussian, axis=-1)  
        return inputs * gaussian

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({'trainable': False})
        return config
