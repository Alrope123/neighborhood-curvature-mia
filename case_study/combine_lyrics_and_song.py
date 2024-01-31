import os
import argparse
import pickle as pkl
import json
import math
import numpy as np
import csv


if __name__ == '__main__':
    lyrics_path = "/gscratch/h2lab/alrope/data/lyrics/lyrics-data.csv"
    outputs_path = "/gscratch/h2lab/alrope/data/lyrics/lyrics-data.jsonl"

    outputs = {}
    with open(lyrics_path, mode='r', encoding='utf-8') as file:
        csv_dict_reader = csv.DictReader(file)
        for row in csv_dict_reader:
            outputs["Artist"] = row["ALink"]
            outputs["text"] = row["Lyric"]

    with open(outputs_path, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output))
            f.write('\n')
            
