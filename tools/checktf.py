import os
import tensorflow as tf
import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(project_dir)

MAX_HEIGHT = 1080
MAX_WIDTH = 1920

input_image_path = f'{project_dir}/data/images/train/movie_1488.jpg'
output_image_path = f'{project_dir}/output.jpg'

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    height, width = tf.shape(image)[0], tf.shape(image)[1]

    # the same
    if height == MAX_HEIGHT and width == MAX_WIDTH:
        return image

    # smaller - only pad
    if height <= MAX_HEIGHT and width <= MAX_WIDTH:
        image = tf.image.resize_with_crop_or_pad(image, MAX_HEIGHT, MAX_WIDTH)
        return image

    # bigger in 1 or 2 dims - resize and pad
    image = tf.image.resize_with_pad(image, MAX_HEIGHT, MAX_WIDTH)
    return image

def save_image(image, output_path):
    image = tf.cast(image * 255.0, tf.uint8)
    encoded_image = tf.image.encode_jpeg(image)
    tf.io.write_file(output_path, encoded_image)

def main():
    image = load_image(input_image_path)
    save_image(image, output_image_path)

if __name__ == '__main__':
    main()