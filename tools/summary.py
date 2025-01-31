import os, sys, json
import tensorflow as tf
from tensorflow.keras.utils import plot_model

project_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(project_dir)
sys.path.append(f'{project_dir}/src')

import models

def main():
    model = models.init_model_continuous(512, 512)

    plot_model(model, to_file="arch.png", show_shapes=True, show_dtype=True, show_layer_names=True, show_trainable=True)

    model.summary()
    print(f"Optimizer: \n{json.dumps(model.optimizer.get_config(), indent=4)}")
    print(f"Loss function: {model.loss.name}")
    print(f"Metrics: {[metric.name for metric in model.metrics]}")
    print(f"Input shape {model.input_shape}")
    print(f"Output shape {model.compute_output_shape(model.input_shape)}")

    # print(f"\nlayers: {model.to_json(indent=2)}")

if __name__ == '__main__':
    main()