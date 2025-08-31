import csv
import random

def load_data(filepath):
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["HeartDisease"] = int(row["HeartDisease"])
            rows.append(row)
    return rows

def train_test_split(rows, test_size=0.2, shuffle=True):
    if shuffle:
        random.shuffle(rows)
    split = int((1 - test_size) * len(rows))
    return rows[:split], rows[split:]
