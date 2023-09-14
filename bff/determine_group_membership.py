import json
import numpy as np
import os
# from scipy.stats import entropy
import pickle as pkl
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

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
  

def decide_member(is_members, group_threshold):
    return sum(is_members) / len(is_members) > group_threshold 


def main(args):
    # Process args
    save_dir = args.save_dir
    data_type = args.data_type
    document_threshold = args.document_threshold
    group_threshold = args.group_threshold
    assert os.path.exists(save_dir)
    save_dir = os.path.join(save_dir, data_type)
    assert os.path.exists(save_dir)

    membership_info_path = os.path.join(save_dir, 'group_to_member.pkl')
    with open(membership_info_path, "rb") as f:
        membership_info = pkl.load(f)

    for group, infos in membership_info.items():
        is_members = []
        for j, (filename, i, score) in enumerate(infos['documents']):
            is_member = score > document_threshold
            membership_info[group]['documents'][j] = (filename, i, score, is_member)
            is_members.append(is_member)
        membership_info[group]['is_members'] = is_members
        membership_info[group]['group_is_member'] = decide_member(is_members, group_threshold)

    # Create statistic info
    print("Calculating the statistics...")
    group_lengths = [len(infos['is_members']) for _, infos in membership_info.items()]
    group_rates = {group: sum(infos['is_members']) / len(infos["is_members"]) for group, infos in membership_info.items()}
    stats = {
        "document_threshold": document_threshold,
        "group_threshold": group_threshold,
        "number of groups": len(membership_info),
        "number of member group": sum([infos['group_is_member'] for _, infos in membership_info.items()]),
        "number of non-member group": sum([not infos['group_is_member'] for _, infos in membership_info.items()]),
        "average number of documents in every group": np.mean(group_lengths),
        "std number of documents in every group": np.std(group_lengths),
        "groups": group_rates
    }
    with open(os.path.join(save_dir, 'grouping_stats.json'), "w") as f:
        json.dump(stats, f)
    draw_histogram(group_lengths, title=None, xlabel="# documents each date", ylabel="# Dates(k)",
                    save_path=os.path.join(save_dir, 'documents_date_distribution.png'), bins=50)
    draw_histogram(list(group_rates.values()), title=None, xlabel="Percentage of Members", ylabel="# Dates(k)",
                    save_path=os.path.join(save_dir, 'membership_distribution.png'), bins=20, x_interval=0.05)

    with open(membership_info_path, "wb") as f:
        pkl.dump(membership_info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv")
    parser.add_argument('--data_type', type=str, default="rpj-arxiv")
    parser.add_argument('--document_threshold', type=float, default="0.1")
    parser.add_argument('--group_threshold', type=float, default="0.1")

    args = parser.parse_args()

    main(args)
