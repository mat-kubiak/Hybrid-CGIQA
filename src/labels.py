import csv, os
import numpy as np

def load_data(mos_path, image_dir):
    with open(mos_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        parsed_data = [row for row in reader]

    rows = parsed_data[1:]

    imgpaths = np.array(sorted(os.listdir(image_dir)))
    filtered_rows = [row for row in rows if row[0] in imgpaths]

    data = np.array([
        [os.path.join(image_dir, row[0]), np.float32(row[1]) / 5.0]  # Normalize MOS to [0, 1]
        for row in filtered_rows
    ])

    return data