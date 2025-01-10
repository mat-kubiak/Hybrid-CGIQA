import os, sys, tqdm
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images

dataset_filename = 'dataset'

max_width = 1280
max_height = 720

dataset_dir = f'{project_dir}/data/raw'

fragment_size = 50
fragment_range = range(0, 5)

def main():
    imgpaths = os.listdir(dataset_dir + '/images')
    imgnum = len(imgpaths)
    print(f"detected images: {imgnum}")

    for i in fragment_range:
        
        x_train = []
        load_range = range(i*fragment_size, (i+1)*fragment_size)
        
        for i in tqdm.tqdm(load_range):
            image = images.load_img(f'{dataset_dir}/images/{imgpaths[i]}')
            
            resized = images.resize_image(image, max_width, max_height)
            # resized = images.pad_image(image, max_width, max_height)

            x_train.append(resized)

        x_train = np.array(x_train, dtype=np.float16)

        np.save(f'{project_dir}/data/processed/{dataset_filename}_{load_range.start}-{load_range.stop}.npy', x_train)

if __name__ == '__main__':
    main()
