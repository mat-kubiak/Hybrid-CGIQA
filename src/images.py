import os
import tensorflow as tf
import numpy as np

def load_img(path, height, width):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    
    image = tf.image.resize_with_pad( image, height, width, method=ResizeMethod.BILINEAR, antialias=False)
    
    return image

def get_image_list(dir_path):
    filenames = np.sort(np.array(os.listdir(dir_path)))
    return dir_path + "/" + filenames
