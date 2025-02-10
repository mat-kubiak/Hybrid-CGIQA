import os, sys

VENDOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vendor')
WEIGHTS_PATH = os.path.join(VENDOR_PATH, 'weights_mobilenet_aesthetic_0.07.hdf5')

sys.path.append(VENDOR_PATH)
from nima_model_builder import Nima

def load_pretrained_nima():
    nima = Nima(base_model_name="MobileNet", weights=None)
    nima.build()

    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f'Weights for NIMA not found at: {WEIGHTS_PATH}')

    nima.nima_model.load_weights(WEIGHTS_PATH)

    return nima.nima_model
