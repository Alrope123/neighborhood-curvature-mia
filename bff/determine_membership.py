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

member_dict_path = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/pile_member_text_w_time.pkl"
nonmember_dict_path =  "/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/pile_nonmember_text_w_time.pkl"
member_dict = {}
nonmember_dict = {}

member_dict_path2 = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out2/pile_member_text_w_time.pkl"
nonmember_dict_path2 =  "/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out2/pile_nonmember_text_w_time.pkl"
member_dict2 = {}
nonmember_dict2 = {}

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

instruction_v1_set = ["sharegpt", "flan_v2", "cot", "gpt4_alpaca", "oasst1", "code_alpaca", "dolly"]
instruction_v2_set = ['code_alpaca', 'hard_coded', 'science.scierc_ner', 'cot', 'wizardlm', 'science.qasper_truncated_4000', 'open_orca', 'lima', 'science.scierc_relation', 'gpt4_alpaca', 'oasst1', 'science.scifact_json', 'flan_v2', 'science.evidence_inference', 'science.scitldr_aic', 'sharegpt']
def custom_open_yield(path, suffix=".jsonl"):
    print("Suffix={}".format(suffix))
    if suffix == ".jsonl":
        with open(path, 'r') as file:
            for line in file:
                dp = json.loads(line)
                yield dp
    elif suffix == "huggingface":
        def filter_rows(row):
            # Replace 'value_to_delete' with the value which, if found, will lead to row deletion
            return row['dataset'] not in instruction_v1_set
        dataset_v1 = load_dataset("allenai/tulu-v1-sft-mixture", split="train")
        dataset_v2 = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
        dataset_v2 = dataset_v2.filter(filter_rows)
        merged_dataset = concatenate_datasets([dataset_v1, dataset_v2])
        for dp in merged_dataset:
            yield dp
    elif suffix == "csv":
        with open(path, mode='r', encoding='utf-8') as file:
            csv_dict_reader = csv.DictReader(file)
            for row in csv_dict_reader:
                # Each row is a dictionary
                yield row
    else:
        raise NotImplementedError()


def calculate_coverage(dp):
    return dp['bff_contained_ngram_count'] / dp['bff_ngram_count'] if dp['bff_ngram_count'] > 0 else 0


title_count = 0
short_title_count = 0
def get_group(dp, data_type):
    global member_dict
    global nonmember_dict
    global member_dict2
    global nonmember_dict2

    if data_type == 'rpj-arxiv_month':
        timestamp = dp['meta']['timestamp']
        assert 'T' in timestamp
        timestamp = timestamp.split('T')[0]
        timestamp_splits = timestamp.split('-')
        timestamp = '-'.join(timestamp_splits[:-1])
        return timestamp
    elif data_type.startswith('rpj-arxiv'):
        timestamp = dp['meta']['timestamp']
        assert 'T' in timestamp
        return timestamp.split('T')[0]
    elif data_type == 'wikipedia_month':
        if member_dict == {}:
            with open(member_dict_path, 'rb') as f:
                member_dict = pkl.load(f)
                print("Loaded in {} wikipedia titled matched.".format(len(member_dict)))
        if nonmember_dict == {}:
            with open(nonmember_dict_path, 'rb') as f:
                nonmember_dict = pkl.load(f)
                print("Loaded in {} wikipedia titled unmatched.".format(len(nonmember_dict)))
        title = dp['title']
        timestamp = None
        if title in member_dict and member_dict[title] != None:
            timestamp = datetime.strptime(member_dict[title].split(',')[1].strip(), '%d %B %Y').strftime('%Y-%m-%d')
            timestamp_splits = timestamp.split('-')
            timestamp = '-'.join(timestamp_splits[:-1])
        elif title in nonmember_dict and nonmember_dict[title] != None:
            timestamp = datetime.strptime(nonmember_dict[title].split(',')[1].strip(), '%d %B %Y').strftime('%Y-%m-%d')
            timestamp_splits = timestamp.split('-')
            timestamp = '-'.join(timestamp_splits[:-1])
        return timestamp 
    elif data_type.startswith('wikipedia_anchor'):
        if member_dict2 == {}:
            with open(member_dict_path2, 'rb') as f:
                member_dict2 = pkl.load(f)
                print("Loaded in {} wikipedia titled matched.".format(len(member_dict2)))
        if nonmember_dict2 == {}:
            with open(nonmember_dict_path2, 'rb') as f:
                nonmember_dict2 = pkl.load(f)
                print("Loaded in {} wikipedia titled unmatched.".format(len(nonmember_dict2)))
        title = dp['title']
        if title in member_dict2 and member_dict2[title] != None:
            return datetime.strptime(member_dict2[title].split(',')[1].strip(), '%d %B %Y').strftime('%Y-%m-%d')
        elif title in nonmember_dict2 and nonmember_dict2[title] != None:
            return datetime.strptime(nonmember_dict2[title].split(',')[1].strip(), '%d %B %Y').strftime('%Y-%m-%d')
        else:
            return None
    elif data_type.startswith('wikipedia'):
        if member_dict == {}:
            with open(member_dict_path, 'rb') as f:
                member_dict = pkl.load(f)
                print("Loaded in {} wikipedia titled matched.".format(len(member_dict)))
        if nonmember_dict == {}:
            with open(nonmember_dict_path, 'rb') as f:
                nonmember_dict = pkl.load(f)
                print("Loaded in {} wikipedia titled unmatched.".format(len(nonmember_dict)))
        title = dp['title']
        if title in member_dict and member_dict[title] != None:
            return datetime.strptime(member_dict[title].split(',')[1].strip(), '%d %B %Y').strftime('%Y-%m-%d')
        elif title in nonmember_dict and nonmember_dict[title] != None:
            return datetime.strptime(nonmember_dict[title].split(',')[1].strip(), '%d %B %Y').strftime('%Y-%m-%d')
        else:
            return None
    elif data_type.startswith("rpj-book_author"):
        if 'title' in dp['meta']:
            splits = dp['meta']['title'].split(" - ")
            if len(splits) != 2:
                print("Error: {}".format(splits)) 
                return None
            return splits[1]
        elif 'short_book_title' in dp['meta']:
            splits = dp['meta']['short_book_title'].split(" by ")
            if len(splits) != 2:
                print("Error: {}".format(splits))
                return None 
            return splits[1]
        else:
            raise NotImplementedError("Key not in the meta")
    elif data_type.startswith("rpj-book"):
        if 'title' in dp['meta']:
            return "Books3-" + dp['meta']['title']
        elif 'short_book_title' in dp['meta']:
            return "Gutenberg-" + dp['meta']['short_book_title']
        else:
            raise NotImplementedError("Key not in the meta")
    elif data_type.startswith("language"):
        return dp["meta"]["language"]
    elif data_type.startswith("instruction"):
        return dp["dataset"]
    elif data_type.startswith('license'):
        return dp["subset_name"]
    elif data_type.startswith('lyrics'):
        return dp["Artist"]
    elif data_type.startswith('nytimes'):
        return dp["date"].split()[0]
    else:
        raise NotImplementedError('The data type is not implemented yet.')


def qualified(data_type, score=None, group=None):
    if data_type=='wikipedia':
        return (score < 0.05 and group >= "2020-03-01") or (score > 0.95 and group < "2020-03-01")
    elif data_type=="wikipedia_anchor":
        return group < "2021-7-18" or group >= "2023-7-18"
    else:
        return True


def get_wikipedia_label(dp):
    global member_dict
    global nonmember_dict

    if member_dict == {}:
        with open(member_dict_path, 'rb') as f:
            member_dict = pkl.load(f)
            print("Loaded in {} wikipedia titled matched.".format(len(member_dict)))
    if nonmember_dict == {}:
        with open(nonmember_dict_path, 'rb') as f:
            nonmember_dict = pkl.load(f)
            print("Loaded in {} wikipedia titled unmatched.".format(len(nonmember_dict)))
    title = dp['title']
    if title in member_dict and member_dict[title] != None:
        return "Title Matched"
    elif title in nonmember_dict and nonmember_dict[title] != None:
        return "Title Unmatched"
    else:
        return None


def draw_histogram(data, save_path, bins=None, title=None, xlabel=None, ylabel=None, cumulative=False, x_interval=-1):
    """Draw a histogram for the given data."""
    
    plt.clf()
    plt.figure(figsize=(10,6))  # Set the figure size
    plt.hist(data, bins=bins, color='#1b9e77', edgecolor=None, density=True, cumulative=cumulative)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if x_interval > 0:
        x_ticks = np.arange(min(data), max(data) + x_interval, x_interval)
        plt.xticks(x_ticks)
    
    # plt.grid(axis='y', alpha=0.75)  # Add a grid on y-axis
    plt.savefig(save_path, format='png') 


def draw_separate_histogram(coverages, split=None, bins=20, xlabel=None, ylabel=None, save_path=None):    
    plt.clf()

    # Separate the numbers and properties
    values, categories = zip(*coverages)

    # Merge categories
    if split:
        categories = np.searchsorted(split, categories, side='right')
        assert all([category >= 0 and category <= len(split) for category in categories])
        categories = ["<{}".format(split[i]) for i in categories]
    elif split == []:
        categories = ["All" for _ in categories]

    # Define bin edges
    bin_edges = np.linspace(min(values), max(values), bins+1)  # Example: 20 bins
    bin_edges[-1] = bin_edges[-1] + 1e-10
    binned_values = np.digitize(values, bin_edges)

    # Prepare data for stacked bars
    bin_counts = {i: {cat: 0 for cat in set(categories)} for i in range(1, len(bin_edges))}

    for bv, cat in zip(binned_values, categories):
        bin_counts[bv][cat] += 1

    # Create gradient colors based on the number of unique categories
    unique_categories = sorted(list(set(categories)))
    colormap = plt.get_cmap('viridis')
    colors = [colormap(i) for i in np.linspace(0, 1, len(unique_categories))]

    # Plotting
    bottoms = [0] * (len(bin_edges) - 1)
    for idx, cat in enumerate(unique_categories):
        cat_counts = [bin_counts[i][cat] for i in range(1, len(bin_edges))]
        plt.bar(range(1, len(bin_edges)), cat_counts, color=colors[idx], label=cat, bottom=bottoms)
        bottoms = [i+j for i, j in zip(bottoms, cat_counts)]

    # Setting x-tick labels to represent bin ranges
    tick_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
    plt.xticks(range(1, len(bin_edges)), tick_labels, rotation=45, ha="right")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path, format='png')

rpj_pile_member_set = ["pt", "en", "fr", "ca", "es"]
rpj_pile_nonmember_set = ["sv", "da", "ro", "bg", "pl", "hu", "sl", "uk", "cs", "it", "ru", "sr", "nl", "de", "hr"]
def main(args):
    # Process args
    data_dir = args.data_dir
    overlap_dir = args.overlap_dir
    save_dir = args.save_dir
    data_type = args.data_type
    read_cache = args.read_cache
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, data_type)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filter_names = os.listdir(overlap_dir) if overlap_dir else [0, 1]   


    membership_info_path = os.path.join(save_dir, 'group_to_member.pkl')
    if not os.path.exists(membership_info_path) or not read_cache:
        # Process each file
        print("Going through each file to check BFF results...")
        membership_info = {}
        for i, (data_path, filename) in enumerate(tqdm(iterate_files(data_dir))):
            # DEBUG:
            # if i > 3:
            #     break

            coverage_path = os.path.join(save_dir, '{}.pkl'.format(filename))
            if os.path.exists(coverage_path):
                print("?????????????????????????")
                with open(coverage_path, 'rb') as f:
                    total_coverages = pkl.load(f)
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!")
                # is_member_all = [False] * len(data)
                total_coverages = []
                for filter_name in filter_names:  
                    if overlap_dir:
                        overlap_path = os.path.join(overlap_dir, filter_name, filename)
                        # Read in each overlap file
                        overlap_data = custom_open(overlap_path)
                        # assert len(data) == len(overlap_data)
                        total_coverages.append([calculate_coverage(dp) for dp in overlap_data])
                    else:
                        total_coverages.append([1.0] * 9999999)

                assert len(total_coverages) == len(filter_names)
                # assert len(total_coverages[0]) == len(data)
                # aggreate the scores over the filters
                total_coverages = [max(sublist[j] for sublist in total_coverages) for j in range(len(total_coverages[0]))]
                # total_coverages = {(filename, i): (total_coverages[i], get_group(data[i], data_type=data_type)) for i in range(len(total_coverages))}

                total_coverages_dict = {}
                for j, dp in enumerate(custom_open_yield(data_path, suffix=".jsonl")):
                    total_coverages_dict[(filename, j)] = (total_coverages[j], get_group(dp, data_type=data_type))
                total_coverages = total_coverages_dict

                coverage_dir = os.path.dirname(coverage_path)
                if not os.path.exists(coverage_dir):
                    os.makedirs(coverage_dir)
                # Save the coverage information for the file
                with open((coverage_path), "wb") as f:
                    pkl.dump(total_coverages, f)

            # Build group dict and filter out unwanted ones
            for (filename, j), (score, group) in total_coverages.items():
                if group and qualified(data_type, score, group):
                    if group not in membership_info:
                        membership_info[group] = {'documents': []}    
                    membership_info[group]['documents'].append((filename, j, score))

        # Save the membership info
        with open(membership_info_path, "wb") as f:
            pkl.dump(membership_info, f)

    else:
        with open(membership_info_path, "rb") as f:
            membership_info = pkl.load(f)

    coverages_and_group = []
    for group, infos in membership_info.items():
        for (_, _, score) in infos['documents']:
            coverages_and_group.append((score, group))

    if data_type.startswith("rpj-arxiv"):
        draw_separate_histogram(coverages_and_group, split=["1960", "2010", "2020-07-32", "2024"], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=20)
    elif data_type.startswith("wikipedia_anchor"):
        draw_separate_histogram(coverages_and_group, split=[], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=20)
    elif data_type.startswith("wikipedia"):
        draw_separate_histogram(coverages_and_group, split=["1960", "2010", "2020-03-01", "2024"], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=20)
        # draw_separate_histogram(total_coverage_member_values, xlabel="Percentage of duplication", ylabel="# Documents(k)",
        #                             save_path=os.path.join(save_dir, 'overlap_distribution2.png'), bins=20)
    elif data_type.startswith("lyrics"):
        pass
    elif data_type.startswith("nytimes"):
        pass
    elif data_type.startswith("rpj-book"):
        draw_separate_histogram(coverages_and_group, split=[], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=20)
    elif data_type.startswith("language"):
        coverages_and_group_new = []
        for coverage, group in coverages_and_group:
            coverages_and_group_new.append((coverage, group in rpj_pile_member_set))
        coverages_and_group = coverages_and_group_new
        draw_separate_histogram(coverages_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=20) 
    elif data_type.startswith("instruction"):
        draw_separate_histogram(coverages_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=20)
    elif data_type.startswith("license"):
        draw_separate_histogram(coverages_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=20)
    else:
        raise NotImplementedError('The data type is not implemented yet.')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/alrope/data/redpajama/arxiv/")
    parser.add_argument('--overlap_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/")
    parser.add_argument('--data_type', type=str, default="rpj-arxiv")
    parser.add_argument('--read_cache', action="store_true", default=False)

    args = parser.parse_args()

    main(args)
