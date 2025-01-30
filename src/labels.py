import csv, os
import numpy as np

def _load_data(mos_path, image_dir):
    with open(mos_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        parsed_data = [row for row in reader]

    rows = np.array(parsed_data[1:])
    rows = rows[np.argsort(rows[:, 0])]

    imgpaths = sorted(os.listdir(image_dir))
    filtered_rows = np.array([row for row in rows if row[0] in imgpaths])

    mos = filtered_rows[:, 1]
    return mos.astype(np.float32)

def load_ordinal(mos_path, image_dir):
    mos = _load_data(mos_path, image_dir)
    rescaled = np.round(mos / 5.0 * 4.0 + 1, 1)

    cats = np.arange(1.0, 5.1, 0.05)
    ordinal_encoded = np.searchsorted(cats, rescaled)

    return ordinal_encoded

def load_categorical(mos_path, image_dir):
    mos = _load_data(mos_path, image_dir)
    rescaled = np.round(mos / 5.0 * 4.0 + 1, 1)

    cats = np.arange(1.0, 5.1, 0.1)
    ordinal_encoded = np.searchsorted(cats, rescaled)

    one_hot = np.eye(len(cats))[ordinal_encoded]

    return one_hot

def load_continuous(mos_path, image_dir):
    mos = _load_data(mos_path, image_dir)

    normalized = mos / 5.0
    return normalized

def load(mos_path, image_dir, is_categorical):
    if is_categorical:
        return load_categorical(mos_path, image_dir)
    return load_continuous(mos_path, image_dir)