import json
import numpy as np
import os
# from scipy.stats import entropy
import pickle as pkl
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
import csv

cache_dir = "cache"
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, "transformers")

def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)


def custom_open(path, suffix=".jsonl"):
    data = []
    if suffix == ".jsonl":
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))           
    else:
        raise NotImplementedError()
    return data

def custom_open_yield(path, suffix=".jsonl"):
    with open(path, 'r') as file:
        for line in file:
            dp = json.loads(line)
            yield dp



def calculate_coverage(dp):
    return dp['bff_contained_ngram_count'] / dp['bff_ngram_count'] if dp['bff_ngram_count'] > 0 else 0


def qualified(score=None):
    return score > 0.95 


def main(args):
    # Process args
    data_dir = args.data_dir
    overlap_dir = args.overlap_dir
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filter_names = os.listdir(overlap_dir) if overlap_dir else [0, 1]   

    member_dict_path = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/pile_member_text.pkl"
    member_dict = {}
    if member_dict == {}:
        with open(member_dict_path, 'rb') as f:
            member_dict = pkl.load(f)
            print("Loaded in {} wikipedia titled matched.".format(len(member_dict)))

    # Process each file
    print("Going through each file to check BFF results...")
    perfect_set = set()
    for i, (data_path, filename) in enumerate(tqdm(iterate_files(data_dir))):
        total_coverages = []
        for filter_name in filter_names:  
            overlap_path = os.path.join(overlap_dir, filter_name, filename)
            # Read in each overlap file
            overlap_data = custom_open(overlap_path)
            # assert len(data) == len(overlap_data)
            total_coverages.append([calculate_coverage(dp) for dp in overlap_data])

        assert len(total_coverages) == len(filter_names)
        total_coverages = [max(sublist[j] for sublist in total_coverages) for j in range(len(total_coverages[0]))]

        for j, dp in enumerate(custom_open_yield(data_path, suffix=".jsonl")):
            title = dp['title']
            score = total_coverages[j]
            
            if title in member_dict and qualified(score):
                perfect_set.add(title)

    print("Size of the resulting set: {}".format(len(perfect_set)))

    # Save the membership info
    with open(os.path.join(save_dir, "perfect_set.pkl"), "wb") as f:
        pkl.dump(perfect_set, f)



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/alrope/data/redpajama/arxiv/")
    parser.add_argument('--overlap_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/")
    parser.add_argument('--data_type', type=str, default="rpj-arxiv")
    parser.add_argument('--read_cache', action="store_true", default=False)

    args = parser.parse_args()

    main(args)
