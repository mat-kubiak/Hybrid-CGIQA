import os
import tensorflow as tf
import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(project_dir)

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def save_image(image, output_path):
    encoded_image = tf.image.encode_jpeg(image)
    tf.io.write_file(output_path, encoded_image)

input_image_path = f'{project_dir}/data/images/test/movie_1488.jpg'
output_image_path = f'{project_dir}/output.jpg'

image = load_image(input_image_path)

target_height = 1080
target_width = 1920

padded_image = tf.image.resize_with_pad(image, target_height, target_width)

save_image(padded_image, output_image_path)
