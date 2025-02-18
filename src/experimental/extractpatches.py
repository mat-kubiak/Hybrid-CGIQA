import tensorflow as tf

class ExtractPatches(tf.keras.layers.Layer):
    """
    Custom layer that slices 4d tensor into equal-size patches along the 2nd and 3rd dimension.
    It works on any dimension shape, and possibly on unfixed size (None, None, None, channels).

    Resulting patches are stacked along the channel dimensions.
    Because the NormalizedHistogram layer can take any number of channels, they can be easily combined to compute histograms of individual patches of images.
    We ended up not using it, because the number of patches can get very high very quickly.
    """
    def __init__(self, grid_size=(4, 4), **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        patch_height = height // self.grid_size[0]
        patch_width = width // self.grid_size[1]

        size = [
            batch_size, 
            self.grid_size[0],
            patch_height, 
            self.grid_size[1],
            patch_width, 
            channels
        ]
        patches = tf.reshape(inputs, size)

        # to (batch_size, patch_height, patch_width, grid_size[0], grid_size[1], channels)
        patches = tf.transpose(patches, perm=[0, 2, 4, 1, 3, 5])

        size = [
            batch_size,
            patch_height,
            patch_width,
            self.grid_size[0] * self.grid_size[1] * channels
        ]
        flattened = tf.reshape(patches, size)

        return flattened

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        if height is not None and width is not None:
            patch_height = height // self.grid_size[0]
            patch_width = width // self.grid_size[1]
        else:
            patch_height = None
            patch_width = None
        out_channels = channels * self.grid_size[0] * self.grid_size[1]
        
        return (batch_size, patch_height, patch_width, out_channels)

    def get_config(self):
        config = super().get_config()
        config.update({'grid_size': self.grid_size, 'trainable': False})
        return config
