import csv, os
import numpy as np

def load_labels(mos_path, images_path):
    with open(mos_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        parsed_data = [row for row in reader]

    header = parsed_data[0]
    rows = parsed_data[1:]

    # pass only labels for images_path images
    imgpaths = np.array(os.listdir(images_path))
    rows = [row for row in rows if row[0] in imgpaths]

    # extract
    mos_values = [float(row[1]) for row in rows]
    mos_values = np.array(mos_values, dtype=np.float32)

    # normalize to (1-5) scale, round
    normalized = np.round(mos_values * 0.8 + 1.0, 1)

    # encode in matrix
    one_hot_encoded = np.array([np.eye(41)[int((num - 1.0) * 10)] for num in normalized])

    return one_hot_encoded