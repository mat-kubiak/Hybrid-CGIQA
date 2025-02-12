import os
import tensorflow as tf
import numpy as np

def load_image(path, height, width, augment_with=None, antialias=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    if augment_with != None:
        image = augment_with(image)
        image = tf.clip_by_value(image, 0.0, 1.0)

    if height != None and width != None:
        image = tf.image.resize_with_crop_or_pad(image, height, width)
    
    return image

def get_image_list(dir_path):
    filenames = np.sort(np.array(os.listdir(dir_path)))
    return dir_path + "/" + filenames
