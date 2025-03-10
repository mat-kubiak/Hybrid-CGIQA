import os, random
import tensorflow as tf
import numpy as np

SEED = 23478
tf.random.set_seed(SEED)
random.seed(SEED)

def load_image(path, height, width):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [height, width])
    return image

def random_crop_image(image, height, width):
    image = tf.image.random_crop(image, [height, width, 3], seed=SEED)
    return image

def get_image_list(dir_path):
    filenames = np.sort(np.array(os.listdir(dir_path)))
    return dir_path + "/" + filenames
