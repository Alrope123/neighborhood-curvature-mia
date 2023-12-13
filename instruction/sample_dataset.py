import json
import numpy as np
import os
import pickle as pkl
import argparse
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset, concatenate_datasets

instruction_v1_set = ["sharegpt", "flan_v2", "cot", "gpt4_alpaca", "oasst1", "code_alpaca", "dolly"]
instruction_v2_set = ['code_alpaca', 'hard_coded', 'science.scierc_ner', 'cot', 'wizardlm', 'science.qasper_truncated_4000', 'open_orca', 'lima', 'science.scierc_relation', 'gpt4_alpaca', 'oasst1', 'science.scifact_json', 'flan_v2', 'science.evidence_inference', 'science.scitldr_aic', 'sharegpt']


if __name__ == '__main__':
    count = {}

    def filter_rows(row):
        # Replace 'value_to_delete' with the value which, if found, will lead to row deletion
        return row['dataset'] not in instruction_v1_set
    dataset_v1 = load_dataset("allenai/tulu-v1-sft-mixture", split="train")
    dataset_v2 = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    dataset_v2 = dataset_v2.filter(filter_rows)
    merged_dataset = concatenate_datasets([dataset_v1, dataset_v2])
    
    for entry in merged_dataset:
        dataset = entry["dataset"]
        if dataset not in count:
            count[dataset] = 0
        count[dataset] = count[dataset] + 1

    print(count) 

        