import random
import datasets
import os
import json
import pickle
from typing import List
from tqdm import tqdm
import numpy as np

DATASETS = ['rpj-arxiv', 'wikipedia', 'wikipedia_noisy', 'rpj-arxiv_noisy', 'rpj-arxiv_month', 'rpj-book']

def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)


def sample_group(membership_info, n_group=100, n_document_per_group=30, train=True):
    groups = set()
    info_list = list(membership_info.items())
    if n_group < 0:
        n_group = len(info_list)
    random.shuffle(info_list)
    for group, infos in info_list:
        if len(groups) >= n_group:
            break
        if infos['group_is_member'] == train and len(infos['is_members']) >= n_document_per_group:
            groups.add(group)
    # assert len(groups) == n_group, (len(groups), n_group)

    selected_data = set()
    for group, infos in membership_info.items():
        if group in groups:
            new_added_data = []
            for filename, i, _, _ in infos['documents']:
                new_added_data.append((filename, i))
            # Oversample the documents to give room for unqualified document
            random.shuffle(new_added_data)
            new_added_data = new_added_data[:int(n_document_per_group * 1.2)]
            selected_data.update(new_added_data)
    # assert len(selected_data) >=  n_group * n_document_per_group
    return selected_data


def load_wikipedia(membership_info, train=True, SAVE_FOLDER=None, n_group=100, n_document_per_group=30):
    data_dir = "/gscratch/h2lab/alrope/data/wikipedia/processed/"
    
    selected_data = sample_group(membership_info, n_group, n_document_per_group, train)
    
    data = [] 
    meta_data = []
    for file_path, filename in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if (filename, i) in selected_data:
                    dp = json.loads(line)      
                    meta_data.append((filename, i))
                    data.append(dp['text'])
    assert len(data) == len(selected_data)
    # with open(os.path.join(SAVE_FOLDER, "wikipedia_{}.json".format("member" if train else "nonmember")), "w") as f:
    #     print("Saving to {}.....".format(os.path.join(SAVE_FOLDER, "wikipedia_{}.json".format("member" if train else "nonmember"))))
    #     json.dump(meta_data, f)
    return data, meta_data


def load_arxiv(membership_info, train=True, SAVE_FOLDER=None, n_group=100, n_document_per_group=30):
    data_dir = "/gscratch/h2lab/alrope/data/redpajama/arxiv/"
    
    selected_data = sample_group(membership_info, n_group, n_document_per_group, train)
    
    data = [] 
    meta_data = []
    for file_path, filename in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if (filename, i) in selected_data:
                    dp = json.loads(line)      
                    meta_data.append((filename, i))
                    data.append(dp['text'])
    assert len(data) == len(selected_data)
    # with open(os.path.join(SAVE_FOLDER, "metadata_{}.json".format("member" if train else "nonmember")), "w") as f:
    #     print("Saving to {}.....".format(os.path.join(SAVE_FOLDER, "metadata_{}.json".format("member" if train else "nonmember"))))
    #     json.dump(meta_data, f)
    return data, meta_data

def load_book(membership_info, train=True, SAVE_FOLDER=None, n_group=100, n_document_per_group=30):
    data_dir = "/gscratch/h2lab/alrope/data/redpajama/book/"
    
    selected_data = sample_group(membership_info, n_group, n_document_per_group, train)
    
    data = [] 
    meta_data = []
    for file_path, filename in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if (filename, i) in selected_data:
                    dp = json.loads(line)      
                    meta_data.append((filename, i))
                    data.append(dp['text'])
    assert len(data) == len(selected_data)
    # with open(os.path.join(SAVE_FOLDER, "metadata_{}.json".format("member" if train else "nonmember")), "w") as f:
    #     print("Saving to {}.....".format(os.path.join(SAVE_FOLDER, "metadata_{}.json".format("member" if train else "nonmember"))))
    #     json.dump(meta_data, f)
    return data, meta_data

def load(name, membership_path, verbose=False, n_group=100, n_document_per_group=30, train=True, SAVE_FOLDER=None):

    if name in DATASETS:
        if verbose:
            print("Loading the dataset: {}...".format(name))
        with open(membership_path, 'rb') as f:
            membership_info = pickle.load(f)
        load_fn = globals()[f'load_{name.split("_")[0].split("-")[-1]}']
        return load_fn(membership_info, n_group=n_group, n_document_per_group=n_document_per_group, train=train, SAVE_FOLDER=SAVE_FOLDER)
    else:
        raise ValueError(f'Unknown dataset {name}')
