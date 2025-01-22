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

    # extract and normalize
    mos_values = [float(row[1]) for row in rows]
    mos_values = np.array(mos_values, dtype=np.float32)

    normalized = mos_values / 5.0

    return normalized