import os
from os import listdir
import numpy as np
import random
import shutil

folder_dir = "./data/images/movies"
new_dir = "./data/images/movies/test"

imgpaths = np.array(os.listdir(folder_dir))

choices = []

# 100% = 3000 all images
# 80% = 2400 train images
# 20% = 600 test images

imgs_left = 600
while (imgs_left > 0):
    choice = random.randint(0, 2999)
    if (choice not in choices):
        choices.append(choice)
        imgs_left -= 1
        

choices = np.sort(np.array(choices))
choices = [f"movie_{choice:04}.jpg" for choice in choices]

for choice in choices:
    old_path = f'{folder_dir}/{choice}'
    new_path = f'{new_dir}/{choice}'
    shutil.move(old_path, new_path)
