import csv

def load_labels(path):
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        parsed_data = [row for row in reader]

    header = parsed_data[0]
    rows = parsed_data[1:]

    mos_values = [float(row[1]) / 5.0 for row in rows]
    return mos_values