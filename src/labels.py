import csv
import numpy as np

def load_labels(path):
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        parsed_data = [row for row in reader]

    header = parsed_data[0]
    rows = parsed_data[1:]

    mos_values = [float(row[1]) for row in rows]
    mos_values = np.array(mos_values, dtype=np.float32)

    # normalize to (1-5) scale, round
    normalized_mos = np.round(mos_values * 0.8 + 1.0, 1)
    
    return normalized_mos