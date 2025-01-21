import os, json
from tensorflow import keras

project_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = f'{project_dir}/model.keras'

model = keras.models.load_model(MODEL_PATH)

print(model.summary())
print(f"Optimizer: \n{json.dumps(model.optimizer.get_config(), indent=4)}")
print(f"Loss function: {model.loss}")
print(f"Metrics: {model.metrics}")
print(f"Input shape {model.input_shape}")