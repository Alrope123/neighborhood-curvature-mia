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

def concatenate_messages(messages):
    text = ""
    for message in messages:
        text += "<|{}|>\n{}\n".format(message["role"], message["content"])
    return text

if __name__ == '__main__':
    np.random.seed(2023)
    random.seed(2023)

    old_selected_indices = {}
    selected_indices = {}
    def filter_rows(row):
        # Replace 'value_to_delete' with the value which, if found, will lead to row deletion
        return row['dataset'] not in instruction_v1_set
    def select_dataset(row):
        selected_datasets = ["oasst1", "dolly", "sharegpt", "gpt4_alpaca", "code_alpaca", "lima", 'science.scierc_ner', 'science.qasper_truncated_4000','science.evidence_inference', 'science.scitldr_aic']
        return row['dataset'] not in selected_datasets
    
    dataset_v1 = load_dataset("allenai/tulu-v1-sft-mixture", split="train")
    dataset_v2 = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    dataset_v2 = dataset_v2.filter(filter_rows)
    merged_dataset = concatenate_datasets([dataset_v1, dataset_v2])
    merged_dataset = merged_dataset.filter(select_dataset)

    dataset_mt_bench = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    dataset_alpaca_bench = load_dataset("tatsu-lab/alpaca_eval", split="eval")
    
    new_dataset = {}

    for entry in merged_dataset:
        dataset = entry["dataset"]
        entry['text'] = entry['messages']
        del entry['messages']
        if dataset not in new_dataset:
            new_dataset[dataset] = []
        new_dataset[dataset].append(entry)

    for entry in dataset_mt_bench:
        dataset = "mt_bench"
        entry['text'] = entry['prompt']
        del entry['prompt']
        if dataset not in new_dataset:
            new_dataset[dataset] = []
        new_dataset[dataset].append(entry)

    for entry in dataset_mt_bench:
        dataset = "alpaca_bench"
        entry['text'] = entry['prompt']
        del entry['prompt']
        if dataset not in new_dataset:
            new_dataset[dataset] = []
        new_dataset[dataset].append(entry)

    for key, value in new_dataset.items():
        with open(os.path.join("/gscratch/h2lab/alrope/data/instruction_v2/{}.jsonl".format(key)), 'w') as f:
            for entry in value:
                f.write(json.dumps(entry) + "\n")


        