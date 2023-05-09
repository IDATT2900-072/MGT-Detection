import os
import csv

from pathlib import Path


def path_to_csv(target_dir_path, target_file_name):
    return target_dir_path + "/" + target_file_name + ".csv"


def create_csv_if_nonexistent(columns, target_dir_path, target_file_name):
    path = path_to_csv(target_dir_path, target_file_name)
    if not Path(path).is_file():
        print("No file already exists. Creating blank CSV\n")
        os.makedirs(target_dir_path, exist_ok=True)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
    else:
        print("CSV-file already exists. Will append new rows to existing document. Cancel execution if this is not "
              "intended.\n")


def write_csv_row(row, path_to_csv):
    with open(path_to_csv, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def write_csv(columns, rows, target_dir_path, target_file_name):
    os.makedirs(target_dir_path, exist_ok=True)
    with open(target_dir_path + "/" + target_file_name + ".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)