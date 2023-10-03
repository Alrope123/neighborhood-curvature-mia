import os
import json
import csv
import numpy as np
from tqdm import tqdm
import pickle as pkl


def split_by_chapter(text):
    # Determine how many Chapter I are there
    idx = max(text.find("CHAPTER I "), text.find("CHAPTER I.") > 0)
    assert idx >= 0
    while idx >= 0:
        text = text[idx: ]
        new_idx = max(text.find("CHAPTER I "), text.find("CHAPTER I.") > 0)
        if new_idx == idx:
            break
        else:
            idx = new_idx
        
    chapters = text.split("CHAPTER")
    chapters = ["CHAPTER" + chapter for chapter in chapters if chapter]
    return chapters


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
    data_dir = "/gscratch/h2lab/alrope/data/redpajama/book/"
    save_dir = "/gscratch/h2lab/alrope/data/redpajama/book-chapters/"
    chapter_keywords = ["CHAPTER I", 'CHAPTER II', 'CHAPTER III']

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for file_path, file_name in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as fin, open(os.path.join(save_dir, file_name), 'w') as fout:
            for line in fin:
                dp = json.loads(line)
                text = dp['text']
                if all([keyword in text for keyword in chapter_keywords]):
                    for chapter in split_by_chapter(text):
                        dp['text'] = chapter
                        fout.write(json.dumps(dp))
                        fout.write('\n')