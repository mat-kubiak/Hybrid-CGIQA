import os, sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(f'{project_dir}/src')

import images

# Path to the dir containing images
IMG_DIRPATH = f'{project_dir}/data/images/test'

# force a concrete image instead of picking at random
IMG_FORCE = None
IMG_FORCE = f"/home/users/s242221/iqa-thesis/data/images/test/movie_1710.jpg"

MAX_HEIGHT = 720
MAX_WIDTH = 1280

def main():
    img_paths = images.get_image_list(IMG_DIRPATH)
    
    path = None
    if IMG_FORCE == None:
        path = f"{IMG_DIRPATH}/{np.random.choice(img_paths)}"
    else:
        path = IMG_FORCE
    
    image = images.load_img(path, MAX_HEIGHT, MAX_WIDTH)

    middle = image.shape[0]//2
    pixels = image.shape[0]*image.shape[1]*image.shape[2]

    np.set_printoptions(threshold=np.inf)

    print(f"middle row: {image[middle, :, 0]}")
    
    print(f"image chosen: {path}")
    print(f"shape: {image.shape}")
    print(f"No. pixels: {pixels}")
    print(f"max: {np.max(image):4f}")
    print(f"min: {np.min(image):4f}")
    print(f"avg: {np.average(image):4f}")
    print(f"pixel<0.01%: {np.sum(np.array(image).flatten().flatten() < 0.01) / pixels * 100:2f}%")

if __name__ == '__main__':
    main()