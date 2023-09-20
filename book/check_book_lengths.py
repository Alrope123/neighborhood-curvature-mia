import os
import json
import csv
import numpy as np
from tqdm import tqdm

def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)


if __name__ == "__main__":
    data_dir = "/gscratch/h2lab/sewon/data/books3/metadata/metadata.jsonl"
    
    lengths = []
    for file_path, file_name in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            n_words = 0
            for line in f:
                n_words += line.split()
            lengths.append(n_words)
    
    print("Total number is {}.".format(len(lengths)))
    print("# Words in documents: {}({})".format(np.mean(lengths), np.std(lengths)))