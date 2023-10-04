import os
import json
import csv
import numpy as np
from tqdm import tqdm
import pickle as pkl
import re

def find_chapter_i_index(text):
    # The regex pattern for "CHAPTER I" not followed by another roman numeral
    pattern = r'\bCHAPTER I\b(?![IVXLCDM])|CHAPTER I[\W_]'
    match = re.search(pattern, text)
    if match:
        return match.start()
    else:
        return -1

def split_by_chapter(text):
    # Determine how many Chapter I are there
    idx = find_chapter_i_index(text)
    if idx < 0:
        print([text[text.find("CHAPTER I"): text.find("CHAPTER I")+ 500]])
        return []
    while idx >= 0:
        print("found match at {}: {}".format(idx, [text[idx: idx+500]]))
        text = text[idx: ]
        new_idx = find_chapter_i_index(text[1:])
        if new_idx == idx:
            break
        else:
            idx = new_idx
        
    chapters = text.split("CHAPTER")
    chapters = ["CHAPTER" + chapter for chapter in chapters if chapter]
    return chapters


if __name__ == "__main__":
    with open("/gscratch/h2lab/alrope/neighborhood-curvature-mia/debug/sample_book.txt", 'r') as f:
        text = "\n".join(f.readlines())
    chapters = split_by_chapter(text)
    # print(chapters[:5])