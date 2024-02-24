import json
import numpy as np
import os
# from scipy.stats import entropy
import pickle as pkl
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

def default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError

def draw_separate_histogram(coverages, split=None, bins=20, xlabel=None, ylabel=None, save_path=None):    
    plt.clf()

    # Separate the numbers and properties
    values, categories = zip(*coverages)

    # Merge categories
    if split:
        categories = np.searchsorted(split, categories, side='right')
        assert all([category >= 0 and category <= len(split) for category in categories])
        categories = ["<{}".format(split[i] if i < len(split) else "Max") for i in categories]
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
  
rpj_pile_member_set = ["pt", "en", "fr", "ca", "es"]
rpj_pile_nonmember_set = ["sv", "da", "ro", "bg", "pl", "hu", "sl", "uk", "cs", "it", "ru", "sr", "nl", "de", "hr"]
instruction_v1_set = ["sharegpt", "flan_v2", "cot", "gpt4_alpaca", "oasst1", "code_alpaca", "dolly"]
instruction_v2_set = ['code_alpaca', 'science.scierc_ner', 'cot', 'wizardlm', 'science.qasper_truncated_4000', 'open_orca', 'lima', 'science.scierc_relation', 'gpt4_alpaca', 'oasst1', 'science.scifact_json', 'flan_v2', 'science.evidence_inference', 'science.scitldr_aic', 'sharegpt']
instruction_human_set = ["flan_v2", "cot", "oasst1", "dolly"]
switch = False
def decide_member_group(average_score, group, data_type):
    global switch
    if data_type.startswith('wikipedia_anchor'):
        return group < "2023-07-18"
    elif data_type.startswith('wikipedia'):
        return group < "2020-03-01"
    elif data_type.startswith('rpj-arxiv'):
        return group < "2020-07-32"
    elif data_type.startswith('@title-rpj-book'):
        return group.split("-")[0] == "Books3"
    elif data_type.startswith('rpj-book'):
        return average_score > 0.5
    elif data_type.startswith("language"):
        if group in rpj_pile_member_set:
            return True
        elif group in rpj_pile_nonmember_set:
            return False
        else:
            raise NotImplementedError("Language not identified")
    elif data_type.startswith("instruction"):
        if data_type.endswith("v1"):
            return group in instruction_v1_set
        elif data_type.endswith("v2"):
            return group != "dolly"
        elif data_type.endswith("human"):
            return group in instruction_human_set
        else:
            target_data_type = data_type.split('+')[-1]
            return group == target_data_type
    elif data_type.startswith("license"):
        if data_type.endswith("ccby"):
            return any([group.startswith(prefix) for prefix in ["ccby", "pd", "sw"]])
        elif data_type.endswith("sw"):
            return any([group.startswith(prefix) for prefix in ["pd", "sw"]])
        elif data_type.endswith("pd"):
            return any([group.startswith(prefix) for prefix in ["pd"]])
        else:
            return False
    elif data_type.startswith("lyrics") or data_type.startswith("nytimes"):
        switch = not switch
        return switch
    elif data_type.startswith("contamination") or data_type.startswith("tuning"):
        switch = not switch
        return switch
    else:
        raise NotImplementedError() 

def decide_member_individual(filename, i, score, document_threshold):
    return score > document_threshold

def main(args):
    # Process args
    save_dir = args.save_dir
    data_type = args.data_type
    document_threshold = args.document_threshold
    assert os.path.exists(save_dir)
    save_dir = os.path.join(save_dir, data_type[data_type.index('-')+1 : ] if '@' in data_type else data_type)
    assert os.path.exists(save_dir), save_dir

    membership_info_path = os.path.join(save_dir, 'group_to_member.pkl')
    with open(membership_info_path, "rb") as f:
        membership_info = pkl.load(f)

    scores_and_group = []
    for group, infos in membership_info.items():
        is_members = []
        scores = []
        try:
            for j, (filename, i, score) in enumerate(infos['documents']):
                scores.append(score)
                is_member = decide_member_individual(filename, i, score, document_threshold)
                membership_info[group]['documents'][j] = (filename, i, score, is_member)
                is_members.append(is_member)
        except:
            for j, (filename, i, score, is_member) in enumerate(infos['documents']):
                scores.append(score)
                is_member = decide_member_individual(filename, i, score, document_threshold)
                membership_info[group]['documents'][j] = (filename, i, score, is_member)
                is_members.append(is_member)
        scores_and_group.append((np.mean(scores), group))
        membership_info[group]['is_members'] = is_members
        membership_info[group]['group_is_member'] = decide_member_group(np.mean(scores), group, data_type)

    # Create statistic info
    print("Calculating the statistics...")
    group_lengths_nonmember = [len(infos['is_members']) for _, infos in membership_info.items() if not infos['group_is_member']]
    group_lengths_member = [len(infos['is_members']) for _, infos in membership_info.items() if infos['group_is_member']]
    group_rates = {group: sum(infos['is_members']) / len(infos["is_members"]) for group, infos in membership_info.items()}
    stats = {
        "document_threshold": document_threshold,
        "number of groups": len(membership_info),
        "number of member group": sum([infos['group_is_member'] for _, infos in membership_info.items()]),
        "number of non-member group": sum([not infos['group_is_member'] for _, infos in membership_info.items()]),
        "average number of documents in member group": np.mean(group_lengths_member),
        "std number of documents in member group": np.std(group_lengths_member),
        "number of member group with over 100 documents": len([n for n in group_lengths_member if n > 100]),
        "average number of documents in non-member group": np.mean(group_lengths_nonmember),
        "std number of documents in non-member group": np.std(group_lengths_nonmember),
        "number of non-member group with over 100 documents": len([n for n in group_lengths_nonmember if n > 100]),
        # "groups": group_rates
    }
    print(stats)
    with open(os.path.join(save_dir, 'grouping_stats.json'), "w") as f:
        json.dump(stats, f, default=default, indent=4)
    draw_histogram(group_lengths_member + group_lengths_nonmember, title=None, xlabel="# documents each date", ylabel="# Dates(k)",
                    save_path=os.path.join(save_dir, 'documents_date_distribution.png'), bins=50)
    draw_histogram(list(group_rates.values()), title=None, xlabel="Percentage of Members", ylabel="# Dates(k)",
                    save_path=os.path.join(save_dir, 'membership_distribution.png'), bins=20, x_interval=0.05)
    if data_type.startswith("rpj-arxiv"):
        draw_separate_histogram(scores_and_group, split=["1960", "2010", "2020-07-32", "2024"], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("wikipedia_anchor"):
        draw_separate_histogram(scores_and_group, split=["1960", "2021-07-18", "2023-07-18", "2024"], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("wikipedia"):
        draw_separate_histogram(scores_and_group, split=["1960", "2010", "2020-03-01", "2024"], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith('@title-rpj-book'):
        draw_separate_histogram(scores_and_group, split=["Books3", "Gutenberg"], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("rpj-book"):
        draw_separate_histogram(scores_and_group, split=[], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("instruction"):
        draw_separate_histogram(scores_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("license"):
        draw_separate_histogram(scores_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("lyrics"):
        draw_separate_histogram(scores_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("contamination"):
        draw_separate_histogram(scores_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("tuning"):
        draw_separate_histogram(scores_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    elif data_type.startswith("nytimes"):
        draw_separate_histogram(scores_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    if data_type.startswith("language"):
        scores_and_group_new = []
        for score, group in scores_and_group:
            scores_and_group_new.append((score, group in rpj_pile_member_set))
        scores_and_group = scores_and_group_new
        draw_separate_histogram(scores_and_group, split=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                save_path=os.path.join(save_dir, 'group_bff_distribution.png'), bins=20)
    with open(membership_info_path, "wb") as f:
        pkl.dump(membership_info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv")
    parser.add_argument('--data_type', type=str, default="rpj-arxiv")
    parser.add_argument('--document_threshold', type=float, default="0.5")

    args = parser.parse_args()

    main(args)
