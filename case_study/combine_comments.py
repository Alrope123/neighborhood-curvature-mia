import os
import argparse
import pickle as pkl
import json
import math
import numpy as np
import csv
import tqdm


if __name__ == '__main__':
    lyrics_path = "/gscratch/h2lab/alrope/data/nytimes/nyt-comments-2020.csv"
    outputs_path = "/gscratch/h2lab/alrope/data/nytimes/nyt-comments-2020.jsonl"

    outputs = {}
    with open(lyrics_path, mode='r', encoding='utf-8') as file:
        csv_dict_reader = csv.DictReader(file)
        for row in tqdm.tqdm(csv_dict_reader, total=4962666):
            outputs["date"] = row["createDate"]
            outputs["text"] = row["commentBody"]

    with open(outputs_path, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output))
            f.write('\n')
            
