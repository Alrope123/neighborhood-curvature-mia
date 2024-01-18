import random
import datasets
import os
import json
import pickle
from typing import List
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset, concatenate_datasets

DATASETS = ['rpj-arxiv', 'wikipedia', 'wikipedia_noisy', 'wikipedia_month', 'rpj-arxiv_noisy', 'rpj-arxiv_month', 'rpj-book', 'language', 'instruction_v1', 'instruction_v2', 'instruction_human', 'instruction+cot', 'instruction+flan_v2', 'instruction+dolly', 'instruction+oasst1', 'instruction+code_alpaca', 'instruction+gpt4_alpaca', 'instruction+sharegpt', "license_ccby", "license_sw", "license_pd"]
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

def calculate_tfidf_similarity(texts):
    # Calculate TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate pairwise cosine similarity (as it is a common measure for TF-IDF)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def select_similar_subset(texts, subset_size, direction, iteration):
    if subset_size > len(texts):
        raise ValueError("Subset size cannot be larger than the list size.")

    # Initialize best score to a high value and best subset as empty
    if direction == "diverse":
        best_score = float('inf')
    else:
        best_score = float('-inf')
    best_subset = []

    # Calculate initial similarity matrix
    similarity_matrix = calculate_tfidf_similarity(texts)

    # Heuristic: start with a random subset and iteratively try to improve it
    current_subset_indices = np.random.choice(len(texts), subset_size, replace=False).tolist()

    for _ in range(iteration):  # Number of iterations can be adjusted
        # Calculate average pairwise similarity for current subset
        current_subset = [texts[i] for i in current_subset_indices]
        current_similarity = calculate_tfidf_similarity(current_subset)
        current_score = np.mean(current_similarity)

        if direction == "diverse" and current_score < best_score:
            best_score = current_score
            best_subset = current_subset_indices.copy()
        elif direction == "similar" and current_score > best_score:
            best_score = current_score
            best_subset = current_subset_indices.copy()

        # Try a random swap to see if it improves the subset
        new_index = np.random.randint(len(texts))
        if new_index not in current_subset_indices:
            swap_index = np.random.choice(current_subset_indices)
            current_subset_indices.remove(swap_index)
            current_subset_indices.append(new_index)

    return [texts[i] for i in best_subset], best_score


def sample_group(membership_info, n_group=100, n_document_per_group=30, train=True, strategy="random", data_dir=None):
    groups = set()
    info_list = list(membership_info.items())
    if n_group < 0:
        n_group = len([group for group, infos in info_list if infos['group_is_member'] == train])
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
            if strategy == "random":
                random.shuffle(new_added_data)
                new_added_data = new_added_data[:int(n_document_per_group * 1.2)]
            else:
                texts = [] 
                print("Loading data to check for similarity.")
                for file_path, filename in tqdm(iterate_files(data_dir)):
                    with open(file_path, 'r') as f:
                        for i, line in enumerate(f):
                            if (filename, i) in new_added_data:
                                dp = json.loads(line)      
                                texts.append(dp['text'])
                direction = strategy.split("_")[0]
                iteration = int(strategy.split("_")[1])
                print("direction: {}\titeration: {}".format(direction, iteration))
                new_added_data, best_score = select_similar_subset(texts, int(n_document_per_group * 1.2), direction, iteration)
                print("The resulting similarity score is : {}".format(best_score))

            selected_data.update(new_added_data)
    print("Sampled {} groups with {} datapoints.".format(len(groups), len(selected_data)))
    # assert len(selected_data) >= n_group * n_document_per_group, len(selected_data)
    return selected_data


def load_my_dataset(membership_info, data_dir=None, train=True, SAVE_FOLDER=None, n_group=100, n_document_per_group=30, strategy="random"): 
    selected_data = sample_group(membership_info, n_group, n_document_per_group, train, strategy=strategy, data_dir=data_dir)
    
    data = [] 
    meta_data = []
    for file_path, filename in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if (filename, i) in selected_data:
                    dp = json.loads(line)      
                    meta_data.append((filename, i))
                    data.append(dp['text'])
    assert len(data) == len(selected_data), (len(data), len(selected_data))
    return data, meta_data


instruction_v1_set = ["sharegpt", "flan_v2", "cot", "gpt4_alpaca", "oasst1", "code_alpaca", "dolly"]
instruction_v2_set = ['code_alpaca', 'science.scierc_ner', 'cot', 'wizardlm', 'science.qasper_truncated_4000', 'open_orca', 'lima', 'science.scierc_relation', 'gpt4_alpaca', 'oasst1', 'science.scifact_json', 'flan_v2', 'science.evidence_inference', 'science.scitldr_aic', 'sharegpt']
instruction_human_set = ["flan_v2", "cot", "oasst1", "dolly"]
def load_dataset_huggingface(membership_info, data_dir=None, train=True, SAVE_FOLDER=None, n_group=100, n_document_per_group=30): 
    selected_data = sample_group(membership_info, n_group, n_document_per_group, train)
    data = [] 
    meta_data = []
    for file_path, filename in tqdm(iterate_files(data_dir)):
        def filter_rows(row):
            # Replace 'value_to_delete' with the value which, if found, will lead to row deletion
            return row['dataset'] not in instruction_v1_set
        
        def concatenate_messages(messages):
            text = ""
            for message in messages:
                text += "<|{}|>\n{}\n".format(message["role"], message["content"])
            return text
        dataset_v1 = load_dataset("allenai/tulu-v1-sft-mixture", split="train")
        dataset_v2 = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
        dataset_v2 = dataset_v2.filter(filter_rows)
        merged_dataset = concatenate_datasets([dataset_v1, dataset_v2])
        for i, dp in enumerate(merged_dataset):
            if (filename, i) in selected_data:
                meta_data.append((filename, i))
                data.append(concatenate_messages(dp['messages']))
    assert len(data) == len(selected_data), (len(data), len(selected_data))
    return data, meta_data


def load(name, data_dir, membership_path, verbose=True, n_group=100, n_document_per_group=30, train=True, SAVE_FOLDER=None, strategy="random"):

    if name in DATASETS:
        if verbose:
            print("Loading the dataset: {}...".format(name))
        with open(membership_path, 'rb') as f:
            membership_info = pickle.load(f)
        # if name.startswith("instruction"):
        #     return load_dataset_huggingface(membership_info, data_dir=data_dir, n_group=n_group, n_document_per_group=n_document_per_group, train=train, SAVE_FOLDER=SAVE_FOLDER)
        # else:
        return load_my_dataset(membership_info, data_dir=data_dir, n_group=n_group, n_document_per_group=n_document_per_group, train=train, SAVE_FOLDER=SAVE_FOLDER, strategy=strategy) 
    else:
        raise ValueError(f'Unknown dataset {name}')
