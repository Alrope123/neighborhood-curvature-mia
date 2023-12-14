import json
import numpy as np
import os
import pickle as pkl
import argparse
import random
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset, concatenate_datasets

instruction_v1_set = ["sharegpt", "flan_v2", "cot", "gpt4_alpaca", "oasst1", "code_alpaca", "dolly"]
instruction_v2_set = ['code_alpaca', 'hard_coded', 'science.scierc_ner', 'cot', 'wizardlm', 'science.qasper_truncated_4000', 'open_orca', 'lima', 'science.scierc_relation', 'gpt4_alpaca', 'oasst1', 'science.scifact_json', 'flan_v2', 'science.evidence_inference', 'science.scitldr_aic', 'sharegpt']


if __name__ == '__main__':
    np.random.seed(2023)
    random.seed(2023)

    selected_indices = {}
    def filter_rows(row):
        # Replace 'value_to_delete' with the value which, if found, will lead to row deletion
        return row['dataset'] not in instruction_v1_set
    dataset_v1 = load_dataset("allenai/tulu-v1-sft-mixture", split="train")
    dataset_v2 = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    dataset_v2 = dataset_v2.filter(filter_rows)
    merged_dataset = concatenate_datasets([dataset_v1, dataset_v2])
    
    for i, entry in enumerate(merged_dataset):
        dataset = entry["dataset"]
        if dataset not in selected_indices:
            selected_indices[dataset] = []
        selected_indices[dataset].append(i)

    for key, value in selected_indices.items():
        if len(value) < 1000:
            continue
        indices = selected_indices[key]
        indices.shuffle()
        selected_indices[key] = indices[:1000]

    new_dataset = []
    for i, entry in enumerate(merged_dataset):
        dataset = entry["dataset"]
        if i in selected_indices[dataset]:
            new_dataset.append(entry)
    
    with open(os.path.join("/gscratch/h2lab/alrope/data/instruction/0.jsonl"), 'w') as f:
        for entry in new_dataset:
            f.write(json.dumps(entry) + "\n")


        