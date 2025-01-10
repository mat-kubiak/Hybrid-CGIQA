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

frag_range = range(100, 150)

def main():
    model = models.load_model(project_dir + '/model.keras')
    model.summary()
    exit()

    image_labels = labels.load_labels(dataset_dir + '/mos.csv')
    imgpaths = os.listdir(dataset_dir + '/images')
    print(f"detected:\nimages: {len(imgpaths)}\nlabels: {len(image_labels)}")

    range_str = f'{frag_range.start}-{frag_range.stop}'

    print('starting with range: ' + range_str)

    x_train = np.load(f'{project_dir}/data/processed/dataset_{range_str}.npy', mmap_mode='r')
    y_train = image_labels[frag_range.start:frag_range.stop]
    y_train = np.array(y_train)

    print(f"x_train type: {type(x_train)} shape: {x_train.shape}, type: {x_train.dtype}")
    print(f"y_train type: {type(y_train)} shape: {y_train.shape}, type: {y_train.dtype}")

    model = models.load_model(project_dir + '/model.keras')
    # model = models.init_model(max_width, max_height)
    model = models.train_model(model, x_train, y_train, epochs, batch_size)
    
    models.save_model(model, project_dir + '/model.keras')

if __name__ == '__main__':
    main()
