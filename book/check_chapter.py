import os
import json
import csv
import numpy as np
from tqdm import tqdm
import pickle as pkl


def split_by_chapter(text):
    text = text[text.rfind("CHAPTER I"): ]
    chapters = text.split("CHAPTER")
    chapters = ["CHAPTER" + chapter for chapter in chapters]
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
    data_dir = "/gscratch/h2lab/alrope/data/redpajama/book"
    chapter_keywords = ["CHAPTER I", 'CHAPTER II', 'CHAPTER III']
    chapters_lengths_book3 = []
    chapters_lengths_gutenberg = []

    book3_titles = set()
    gutenberg_titles = set()

    for file_path, file_name in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for line in f:
                dp = json.loads(line)
                text = dp['text']
                if 'title' in dp['meta']:
                    if all([keyword in text for keyword in chapter_keywords]):
                        chapters_lengths_book3.append(len(split_by_chapter(text)))
                    book3_titles.add(dp['meta']['title'])
                elif 'short_book_title' in dp['meta']:
                    if all([keyword in text for keyword in chapter_keywords]):
                        chapters_lengths_gutenberg.append(len(split_by_chapter(text)))
                    gutenberg_titles.add(dp['meta']['short_book_title'])
    
    intersection_titles = book3_titles.intersection(gutenberg_titles)

    print("Total number of articles in Book3 is {}.".format(len(book3_titles)))
    print("Number of books that has chapter: {}".format(len(chapters_lengths_book3)))
    print("Average number of chapters: {}").format(sum(chapters_lengths_book3) / len(chapters_lengths_book3))
    print("Total number of articles in Gutenberg is {}.".format(len(gutenberg_titles)))
    print("Number of books that has chapter: {}".format(len(chapters_lengths_gutenberg)))
    print("Average number of chapters: {}").format(sum(chapters_lengths_gutenberg) / len(chapters_lengths_gutenberg))
    print("Total number of intersection is {}.".format(len(intersection_titles)))

    with open("/gscratch/h2lab/alrope/neighborhood-curvature-mia/book/intersection_set.pkl", 'w') as f:
        pkl.dump(f)
