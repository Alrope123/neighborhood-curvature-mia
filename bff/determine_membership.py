import json
import numpy as np
import os
# from scipy.stats import entropy
import pickle as pkl
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def custom_open(path, suffix=".jsonl"):
    data = []
    if suffix == ".jsonl":
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))           
    else:
        raise NotImplementedError()
    return data


def calculate_coverage(dp):
    cover_length = sum([x[1] - x[0] for x in dp["bff_duplicate_spans"]])
    total_length = dp["length"]
    return cover_length / total_length


def get_group(dp, data_type):
    if data_type=='rpj-arxiv':
        timestamp = dp['meta']['timestamp']
        assert 'T' in timestamp
        return timestamp.split('T')[0]
    else:
        raise NotImplementedError('The data type is not implemented yet.')


def draw_histogram(data, save_path, bins=None, title=None, xlabel=None, ylabel=None, cumulative=False):
    """Draw a histogram for the given data."""
    
    plt.figure(figsize=(10,6))  # Set the figure size
    plt.hist(data, bins=bins, color='#1b9e77', edgecolor=None, density=True, cumulative=cumulative)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # plt.grid(axis='y', alpha=0.75)  # Add a grid on y-axis
    plt.savefig(save_path, format='png') 


def main(args):
    # Process args
    data_dir = args.data_dir
    overlap_dir = args.overlap_dir
    save_dir = args.save_dir
    data_type = args.data_type
    threshold = args.threshold

    drawn = False
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filter_names = os.listdir(overlap_dir)    

    # Process each file
    print("Going through each file to check BFF results...")
    group_to_member = {}
    for i, filename in enumerate(tqdm(os.listdir(data_dir))):
        # DEBUG:
        if i > 10:
            break

        # Figure out the path
        data_path = os.path.join(data_dir, filename)
        

        # Load in the data
        data = custom_open(data_path)

        is_member_all = [False] * len(data)
        for filter_name in filter_names:
            overlap_path = os.path.join(overlap_dir, filter_name, filename)
            assert os.path.exists(overlap_path)

            # Read in each overlap file
            overlap_data = custom_open(overlap_path)
            assert len(data) == len(overlap_data)
            coverages = [calculate_coverage(dp) for dp in overlap_data]
            # Draw the distribution of overlaps if haven't drawn
            if not drawn:
                draw_histogram(coverages, title=None, xlabel="Percentage of duplication", ylabel="# Documents(k)", save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=50)
                draw_histogram(coverages, title=None, xlabel="Percentage of duplication", save_path=os.path.join(save_dir, 'overlap_distribution_CDF.png'), bins=50, cumulative=True)
                drawn = True
            is_member = [coverage > threshold for coverage in coverages]
            # DEBUG:
            # print("Filter {}:".format(filter_name))
            # print("\taverage coverage: {}".format(np.mean(coverages)))
            # print("\tmembers: {} / {}".format(sum(is_member), len(is_member)))
            # print("\tall members: {} / {}".format(sum(is_member_all), len(is_member_all)))
            assert len(is_member_all) ==  len(is_member)
            is_member_all = [a or b for a, b in zip(is_member_all, is_member)]

        with open(os.path.join(save_dir, '{}_{}.pkl'.format(filename, threshold)), "wb") as f:
            pkl.dump(is_member_all, f)

        for i, dp in enumerate(data):
            group = get_group(dp, data_type)
            if group not in group_to_member:
                group_to_member[group] = []
            group_to_member[group].append((filename, i, is_member_all[i]))

        # DEBUG:
        # print("Total members: {} / {}".format(sum(is_member_all), len(is_member_all)))

    # Create statistic info
    print("Calculating the statistics...")
    group_lengths = [len(members) for _, members in group_to_member.items()]
    group_member_rate = [sum([member[2] for member in members]) / len(members) for group, members in group_to_member.items()]
    stats = {
        "threshold": threshold,
        "number of groups": len(group_to_member),
        "examples of group": list(group_to_member.keys())[:15],
        "number of group with all members": sum([rate == 1.0 for rate in group_member_rate]),
        "number of group with all non-members": sum([rate == 0.0 for rate in group_member_rate]),
        "average number of instance in every group": np.mean(group_lengths),
        "std number of instance in every group": np.std(group_lengths),
    }
    print(stats)
    with open(os.path.join(save_dir, 'stats.json'), "w") as f:
        json.dump(stats, f)
    draw_histogram(group_member_rate, title=None, xlabel="Percentage of Members", ylabel="# Dates(k)", save_path=os.path.join(save_dir, 'memership_distribution.png'), bins=20)

    # Save the membership info
    with open(os.path.join(save_dir, 'group_to_member.pkl'), "wb") as f:
        pkl.dump(group_to_member, f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/alrope/data/redpajama/arxiv/")
    parser.add_argument('--overlap_dir', type=str, default="/gscratch/h2lab/alrope/data/bff/redpajama-arxiv+pile")
    parser.add_argument('--save_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/out")
    parser.add_argument('--data_type', type=str, default="rpj-arxiv")
    parser.add_argument('--threshold', type=float, default="0.1")

    args = parser.parse_args()

    main(args)
