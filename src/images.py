import cv2
import tensorflow as tf

def _load_img_cv(path):
    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0

    return image

def resize_image(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height))

def prepare_img_for_size(image, target_width, target_height):
    h, w, c = image.shape

def pad_image(image, target_width, target_height):
    h, w, c = image.shape

    if h == target_height and w == target_width:
        return image
    
    top = (target_height - h) // 2
    bottom = target_height - h - top
    left = (target_width - w) // 2
    right = target_width - w - left

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image

def _load_img_tf(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize the image (preserves aspect ratio with padding)
    # image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_img(path):
    return _load_img_cv(path)