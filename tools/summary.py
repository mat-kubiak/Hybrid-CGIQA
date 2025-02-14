import os, sys, json
import tensorflow as tf
from tensorflow.keras.utils import plot_model

project_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(project_dir)
sys.path.append(f'{project_dir}/src')

import models

MODEL_PATH = f''

HEIGHT = 224
WIDTH = 224

def main():
    if len(MODEL_PATH) != 0:
        model = models.load_model(MODEL_PATH)
    else:
        model = models.init_model_continuous(HEIGHT, WIDTH)

    model.summary()

    try:
        plot_model(
            model,
            to_file="arch.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_trainable=True
        )
    except Exception as e:
        print(f'Error while plotting the model: {e}')

    print(f"Hidden layer count: {len(model.layers)-2}")
    print(f"Loss function: {model.loss.name}")
    print(f"Metrics: {[metric.name for metric in model.metrics]}")
    print(f"Input shape {model.input_shape}")
    print(f"Output shape {model.compute_output_shape(model.input_shape)}")
    print(f"Optimizer: \n{json.dumps(model.optimizer.get_config(), indent=4)}")

if __name__ == '__main__':
    main()