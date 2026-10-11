import csv
import os

def save_history(history, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(history)