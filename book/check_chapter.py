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
    contains_chapter_book3 = []
    contains_chapter_gutenberg = []

    book3_titles = set()
    gutenberg_titles = set()

    for file_path, file_name in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for line in f:
                dp = json.loads(line)
                text = dp['text']
                if 'title' in dp['meta']:
                    contains_chapter_book3.append(all([keyword in text for keyword in chapter_keywords]))
                    book3_titles.add(dp['meta']['title'])
                elif 'short_book_title' in dp['meta']:
                    contains_chapter_gutenberg.append(all([keyword in text for keyword in chapter_keywords]))
                    gutenberg_titles.add(dp['meta']['short_book_title'])
    
    intersection_titles = contains_chapter_book3.intersection()

    with open("/gscratch/h2lab/alrope/neighborhood-curvature-mia/book/intersection_set.pkl", 'w') as f:
        

    print("Total number of Book3 is {}.".format(len(contains_chapter_book3)))
    print("Number of books that has chapter: {}".format(sum(contains_chapter_book3)))
    print("Total number of Gutenberg is {}.".format(len(contains_chapter_gutenberg)))
    print("Number of books that has chapter: {}".format(sum(contains_chapter_gutenberg)))
    print("Total number of articles in Book3 is {}.".format(len(book3_titles)))
    print("Total number of articles in Gutenberg is {}.".format(len(gutenberg_titles)))
    print("Total number of intersection is {}.".format(len(intersection_titles)))