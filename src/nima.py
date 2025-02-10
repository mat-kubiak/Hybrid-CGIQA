import sys
from pathlib import Path

VENDOR_PATH = Path(__file__).parent.parent / 'vendor'
WEIGHTS_PATH = VENDOR_PATH / 'weights_mobilenet_aesthetic_0.07.hdf5'

sys.path.append(str(VENDOR_PATH))
from nima_model_builder import Nima

def load_pretrained_nima():
    nima = Nima(base_model_name="MobileNet", weights=None)
    nima.build()

    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f'Weights for NIMA not found at: {WEIGHTS_PATH}')

    nima.nima_model.load_weights(str(WEIGHTS_PATH))

    return nima.nima_model
