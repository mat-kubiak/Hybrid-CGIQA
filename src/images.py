import os
import tensorflow as tf
import numpy as np

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, angles=tf.random.uniform([], -0.1, 0.1))

    image = tf.image.random_brightness(image, max_delta=0.2)  # Brightness change up to ±0.2
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Contrast factor [0.8, 1.2]
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Saturation factor [0.8, 1.2]
    image = tf.image.random_hue(image, max_delta=0.05)  # Hue change up to ±0.05

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = image + noise

    image = tfa.image.gaussian_filter2d(image, filter_shape=(3, 3))
    
    return tf.clip_by_value(image, 0.0, 1.0)

def load_img(path, height, width):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    
    image = augment_image(image)

    image = tf.image.resize_with_pad( image, height, width, method=tf.image.ResizeMethod.BILINEAR, antialias=False)
    
    return image

def get_image_list(dir_path):
    filenames = np.sort(np.array(os.listdir(dir_path)))
    return dir_path + "/" + filenames
