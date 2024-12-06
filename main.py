import os, sys, tqdm
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, models

epochs = 1
model_name = 'model'
batch_size = 5

max_width = 1280
max_height = 720

dataset_dir = f'{project_dir}/data/raw'

load_range = (0, 30)

def main():
    model = None
    image_labels = labels.load_labels(dataset_dir + '/mos.csv')
    imgpaths = os.listdir(dataset_dir + '/images')
    print(f"detected:\nimages: {len(imgpaths)}\nlabels: {len(image_labels)}")

    x_train = []
    for i in tqdm.tqdm(range(load_range[0], load_range[1])):
        image = images.load_img(f'{dataset_dir}/images/{imgpaths[i]}')

        resized = images.resize_image(image, max_width, max_height)
        # resized = images.pad_image(image, max_width, max_height)

        x_train.append(resized)

    y_train = image_labels[load_range[0]:load_range[1]]
    
    x_train = np.array(x_train, dtype=np.float16)
    y_train = np.array(y_train, dtype=np.float16)

    print(f"x_train type: {type(x_train)} shape: {x_train.shape}, type: {x_train.dtype}")
    print(f"y_train type: {type(y_train)} shape: {y_train.shape}, type: {y_train.dtype}")


    model = models.init_model(max_width, max_height)
    model = models.train_model(model, x_train, y_train, epochs, batch_size)
    
    models.save_model(model, project_dir + '/model.keras')

if __name__ == '__main__':
    main()
