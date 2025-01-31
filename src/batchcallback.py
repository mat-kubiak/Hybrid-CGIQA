import tensorflow as tf
import models

class BatchCallback(tf.keras.callbacks.Callback):

    def __init__(self, tracker, target_epochs, model_path, batches_per_epoch, **kwargs):
        super().__init__(**kwargs)
        self.tracker = tracker
        self.target_epochs = target_epochs
        self.model_path = model_path
        self.batches_per_epoch = batches_per_epoch

    def on_batch_end(self, batch, logs=None):
        self.tracker.batch = batch + 1
        self.tracker.log(f"Completed batch {self.tracker.batch}/{self.batches_per_epoch} of epoch {self.tracker.epoch}/{self.target_epochs}")

        self.tracker.save_status()
        self.tracker.append_csv_history(logs)
        
        self.tracker.log(f"Saved status and history")

    def on_epoch_end(self, epoch, logs=None):
        self.tracker.epoch = epoch + 1
        self.tracker.batch = 0

        self.tracker.log(f"Completed epoch {self.tracker.epoch}/{self.target_epochs} completed")
        
        self.tracker.save_status()
        models.save_model(self.model, self.model_path)
        
        self.tracker.log(f"Saved model")
