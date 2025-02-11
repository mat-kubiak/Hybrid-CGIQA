import tensorflow as tf

class WeightsHistogramCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(WeightsHistogramCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            # Log histograms for each layer
            for layer in self.model.layers:       
                if layer.trainable and layer.weights:
                    weights = layer.weights[0].numpy()
                    tf.summary.histogram(f"{layer.name}/weights", weights, step=epoch)
                    if len(layer.weights) > 1:  # If layer has biases
                        bias = layer.weights[1].numpy()
                        tf.summary.histogram(f"{layer.name}/bias", bias, step=epoch)

            self.writer.flush()
