import random
import datasets
import os
import json
import pickle
from typing import List
from tqdm import tqdm

DATASETS = ['rpj-arxiv', 'wikipedia']

def iterate_files(root_dir):
        file_paths = []
        file_names = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                file_names.append(file)
        return file_paths, file_names


def sample_group(check_map, n, train=True):
    groups = set()
    random.seed(2023)
    check_list = list(check_map.items())
    random.shuffle(check_list)
    for _, dp in check_list:
        if len(groups) >= n:
            return groups
        if dp['group_is_member'] == train:
            groups.add(dp['group'])
    return groups


def load_wikipedia(data_dir, check_map, train=True, SAVE_FOLDER=None):
    selected_group = sample_group(check_map, 50, train)
    data = [] 
    save_data = []
    file_paths, filenames = iterate_files(data_dir)
    for file_path, filename in tqdm(zip(file_paths, filenames)):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                dp = json.loads(line)
                title = dp['title']
                if title in check_map and check_map[title]['group'] in selected_group:       
                    save_data.append(dp)
                    data.append(dp['text'])
    with open(os.path.join(SAVE_FOLDER, "wikipedia_{}.json".format("member" if train else "nonmember")), "w") as f:
        json.dump(save_data, f)
    return data


def load(name, data_dir, check_map_dir, verbose=False, train=True, SAVE_FOLDER=None):
    if name in DATASETS:
        if verbose:
            print("Loading the dataset: {}...".format(name))
        with open(check_map_dir, 'wb') as f:
            check_map = pickle.load(f)
        load_fn = globals()[f'load_{name}']
        return load_fn(data_dir, check_map, train=train, SAVE_FOLDER=SAVE_FOLDER)
    else:
        raise ValueError(f'Unknown dataset {name}')
