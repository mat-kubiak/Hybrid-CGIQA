# iqa-thesis

## Acknowledgements

This project builds upon the NIMA implementation trained on the AVA dataset, as presented in Image Quality Assessment by Christopher Lennan, Hao Nguyen, and Dat Tran.
Please cite their work if applicable.

Github repository: https://github.com/idealo/image-quality-assessment

## License

- The code written by me is licensed under the [MIT License](licenses/LICENSE).
- The following files were borrowed in an unchanged form from [Image Quality Assessment](https://github.com/idealo/image-quality-assessment) and are licensed under the [Apache 2.0 License](licenses/LICENSE-APACHE):
  - vendor/nima_model_builder.py
  - vendor/utils/losses.py
  - vendor/weights_mobilenet_aesthetic_0.07.hdf5

## Get started

```py
git clone https://github.com/mat-kubiak/iqa-thesis.git
cd iqa-thesis

# install dependencies in a virtual environment
python3 -m venv env
source ./env/bin/activate.sh
pip install -r requirements.txt

# or directly on your system
pip install tensorflow scikit-learn matplotlib flask opencv-python

# launch script
python3 main.py
```