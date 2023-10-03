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
    data_dir = "/gscratch/h2lab/alrope/data/redpajama/book"
    chapter_keywords = ["CHAPTER I", 'CHAPTER II', 'CHAPTER III']
    contains_chapter = []

    for file_path, file_name in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for line in f:
                dp = json.loads(line)
                text = dp['text']
                contains_chapter.append(all([keyword in text for keyword in chapter_keywords]))

    print("Total number is {}.".format(len(contains_chapter)))
    print("Number of books that has chapter: {}".format(sum(contains_chapter)))