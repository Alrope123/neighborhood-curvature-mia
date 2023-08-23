from collections import defaultdict, OrderedDict
import glob
import json
import mmap
import numpy as np
import os
import random
# from scipy.stats import entropy
import sys
import pickle as pkl
import time
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoXTokenizerFast
from datasets import load_dataset

from memership_label import NGramLanguageModeling

def get_text(data, data_type):
    if data_type=='rpj-arxiv':
        return data['text']
    else:
        raise NotImplementedError('The data type is not implemented yet.')

def get_group(data, data_type):
    if data_type=='rpj-arxiv':
        timestamp = data['meta']['timestamp']
        assert 'T' in timestamp
        return timestamp.split('T')[0]
    else:
        raise NotImplementedError('The data type is not implemented yet.')


def label(lm, text, tokenizer):
    prompt_ids = tokenizer.encode(text)
    return lm.find(prompt_ids, return_boolean=True)


def load_tokenizer(tokenizer_type):
    if tokenizer_type=="llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=access_token)
    elif tokenizer_type=="neox":
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    return tokenizer


def load_pool(pool_type, pool_dir):
    if pool_type.startswith("rpj"):
        subset = pool_type.split('-')[1]
        os.environ["RED_PAJAMA_DATA_DIR"] = pool_dir
        if not subset:
            dataset = load_dataset('togethercomputer/RedPajama-Data-1T')
        else:
            dataset = load_dataset('togethercomputer/RedPajama-Data-1T', subset, split='train')
        return dataset


def save_cache(cache_set, cache_dict, cache_set_path, cache_dict_path):
    with open(cache_set_path, 'r') as f:
        pkl.dump(cache_set, f)
    with open(cache_dict_path, 'r') as f:
        pkl.dump(cache_dict, f)


def main(args):
    cache_dir = args.cache_dir
    tokenizer_type = args.tokenizer_type
    pool_type = args.pool_type
    pool_dir = args.pool_dir
    save_steps = args.save_steps

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    cache_set_path = os.path.join(cache_dir, '{}_{}_labeled_set.pkl'.format(pool_type, tokenizer_type))
    cache_dict_path = os.path.join(cache_dir, '{}_{}_labeled_dict.pkl'.format(pool_type, tokenizer_type))
    
    assert os.path.exists(cache_set_path) == os.path.exists(cache_dict_path)
    if os.path.exists(cache_set_path):
        with open(cache_set_path, 'r') as f:
            cache_set = pkl.load(f)
        with open(cache_dict_path, 'r') as f:
            cache_dict = pkl.load(f)
    else:
        cache_set = set()
        cache_dict = {}
    

    # Load the model
    lm = NGramLanguageModeling(cache_dir, force_cache=False, tokenizer_type=tokenizer_type)
    # Load the datapool
    datapool = load_pool(pool_type, pool_dir)
    print("# datapoints to label: {}".format(len(datapool)))
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_type)


    # Label through the datapool
    for i, dp in datapool:
        if i not in cache_set:
            found = label(lm, get_text(dp), tokenizer)
            cache_set.add(i)
            if get_group(dp) not in cache_dict:
                cache_dict[get_group(dp)] = []
            # Dictionary structure: group_key -> (index in the datapool, label)
            cache_dict[get_group(dp)].append((i, found))

            # Save cache 
            if i % save_steps == 0:
                save_cache(cache_set, cache_dict, cache_set_path, cache_dict_path)

    save_cache(cache_set, cache_dict, cache_set_path, cache_dict_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--tokenizer_type', type=str, default="gpt2")
    parser.add_argument('--pool_type', type=str, default="rpj-arxiv")
    parser.add_argument('--pool_dir', type=str, default="/gscratch/h2lab/alrope/data/redpajama")
    parser.add_argument('--save_steps', type=int, default=1000)

    args = parser.parse_args()

    main(args)
