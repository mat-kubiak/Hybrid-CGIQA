# Hybrid-CGIQA

A lightweight NR-IQA model for CGIs based on the CGIQA-6K dataset and a hybrid architecture.

This code and the published model is a part the engineering thesis "Subjective Methods for Image Quality Assessment" at Lodz University of Technology.

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

## Get Started

First, clone this repository and install dependencies:
```py
git clone https://github.com/mat-kubiak/iqa-thesis.git
cd iqa-thesis

# install dependencies in a virtual environment
python3 -m venv env
source ./env/bin/activate.sh
pip install -r requirements.txt

# or directly on your system
pip install tensorflow scikit-learn matplotlib flask opencv-python
```

Afterwards, images from CGIQA-6K should be placed in the `data/images/` directory.
In case of using another dataset of choice, the file `data/mos.csv` should also be replaced by one containing MOS scores of the new dataset.

> Unfortunately we cannot include those images here, as the whole dataset is 2.5GB in size. Please refer for it [here](https://github.com/zzc-1998/CGIQA6K).

After providing all required data, run these scripts:

script in order to randomly split the dataset:
```py
# randomly split the data (required only once)
python3 tools/split_db.py

# you can customize the training script up to the `# USER PARAMS END HERE` comment
vim train.py

# train the network in background
nohup python3 train.py

# open tensorboard in the background for real-time monitoring
nohup tensorboard --logdir=output --reload_interval=1 --window_title="Hybrid-CGIQA TensorBoard" --port=6006 > tensorboard-nohup.out 2>&1 &

# test your network
vim test.py
python3 test.py
```
