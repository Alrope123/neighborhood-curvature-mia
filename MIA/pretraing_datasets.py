import random
import datasets
import os
import json
import pickle
from typing import List
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, concatenate_datasets

DATASETS = ['rpj-arxiv', 'wikipedia', 'wikipedia_noisy', 'wikipedia_month', 'rpj-arxiv_noisy', 'rpj-arxiv_month', 'rpj-book', 'language', 'instruction_v1', 'instruction_v2', 'instruction_human', 'instruction+cot', 'instruction+flan_v2', 'instruction+dolly', 'instruction+oasst1', 'instruction+code_alpaca', 'instruction+gpt4_alpaca', 'instruction+sharegpt', "license_ccby", "license_sw", "license_pd", "wikipedia_anchor", "lyrics", "nytimes", "contamination", "contamination_full", "tuning"]
cache_dir = "cache"
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, "transformers")

def format_instruct_data(instructions, instruct_model):
    text = ""
    for i, instruction in enumerate(instructions):
        if instruct_model.startswith("allenai/tulu"):
            text += "<|{}|>\n{}\n".format(instruction["role"], instruction["content"])
        elif instruct_model.startswith("WizardLM/WizardLM"):
            text += "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
            if instruction["role"] == "user":
                text += "USER: {} ".format(instruction["content"])
            elif instruction["role"] == "assistant":
                text += "ASSISTANT: {}</s>".format(instruction["content"])
            else:
                raise NotImplementedError("Unrecognized role.")
            text += "\n"
        elif instruct_model.startswith("mistralai/Mistral"):
            text += "<s>"
            if instruction["role"] == "user":
                text += "[INST] {} [/INST]".format(instruction["content"])
            elif instruction["role"] == "assistant":
                text += "{}</s>".format(instruction["content"])
            else:
                raise NotImplementedError("Unrecognized role.")
            text += "\n"
        elif instruct_model.startswith("meta-llama/Llama-2"):
            text += "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
            if instruction["role"] == "user":
                text += "{}{} [/INST] ".format("<s>[INST] " if i != 1 else "", instruction["content"])
            elif instruction["role"] == "assistant":
                text += "{}</s>".format(instruction["content"])
            else:
                raise NotImplementedError("Unrecognized role.")
            text += "\n"
        elif instruct_model.startswith("lmsys/vicuna"):
            text += "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            if instruction["role"] == "user":
                text += "USER: {}\n".format(instruction["content"])
            elif instruction["role"] == "assistant":
                text += "ASSISTANT: {}</s>\n".format(instruction["content"])
            else:
                raise NotImplementedError("Unrecognized role.")
        elif instruct_model.startswith("mine/alpaca"):
            text += "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            if instruction["role"] == "user":
                text += "\n### Instruction: {}\n\n".format(instruction["content"])
            elif instruction["role"] == "assistant":
                text += "### Response: {}\n".format(instruction["content"])
            else:
                raise NotImplementedError("Unrecognized role.")
        else:
            raise NotImplementedError("Unrecognized model.")
    return text


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
            random.shuffle(new_added_data)
            new_added_data = new_added_data[:int(n_document_per_group * 1.2)]
            selected_data.update(new_added_data)
    print("Sampled {} groups with {} datapoints.".format(len(groups), len(selected_data)))
    # assert len(selected_data) >= n_group * n_document_per_group, len(selected_data)
    return selected_data


def load_my_dataset(membership_info, data_dir=None, train=True, SAVE_FOLDER=None, n_group=100, n_document_per_group=30, instruct_model=None): 
    selected_data = sample_group(membership_info, n_group, n_document_per_group, train)
    
    data = [] 
    meta_data = []
    for file_path, filename in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if (filename, i) in selected_data:
                    dp = json.loads(line)      
                    meta_data.append((filename, i))
                    if not instruct_model:
                        data.append(dp['text'])
                    else:
                        data.append(format_instruct_data(dp['text'], instruct_model))
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


def load(name, data_dir, membership_path, verbose=True, n_group=100, n_document_per_group=30, train=True, SAVE_FOLDER=None, instruct_model=None):

    if name in DATASETS:
        if verbose:
            print("Loading the dataset: {}...".format(name))
        with open(membership_path, 'rb') as f:
            membership_info = pickle.load(f)
        # if name.startswith("instruction"):
        #     return load_dataset_huggingface(membership_info, data_dir=data_dir, n_group=n_group, n_document_per_group=n_document_per_group, train=train, SAVE_FOLDER=SAVE_FOLDER)
        # else:
        return load_my_dataset(membership_info, data_dir=data_dir, n_group=n_group, n_document_per_group=n_document_per_group, train=train, SAVE_FOLDER=SAVE_FOLDER, instruct_model=instruct_model) 
    else:
        raise ValueError(f'Unknown dataset {name}')