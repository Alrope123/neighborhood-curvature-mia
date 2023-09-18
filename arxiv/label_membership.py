import json
import numpy as np
import os
# from scipy.stats import entropy
import pickle as pkl
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

member_dict_path = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/pile_member_text_w_time.pkl"
nonmember_dict_path =  "/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/pile_nonmember_text_w_time.pkl"
member_dict = {}
nonmember_dict = {}

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


def calculate_coverage(dp):
    return dp['bff_contained_ngram_count'] / dp['bff_ngram_count'] if dp['bff_ngram_count'] > 0 else 0

def get_group(dp, data_type):
    global member_dict
    global nonmember_dict

    if data_type=='rpj-arxiv':
        timestamp = dp['meta']['timestamp']
        assert 'T' in timestamp
        return timestamp.split('T')[0]
    elif data_type=='wikipedia':
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
    else:
        raise NotImplementedError('The data type is not implemented yet.')


def qualified(data_type, score=None, group=None):
    if data_type == "rpj-arxiv":
        return True
    elif data_type=='wikipedia':
        return (score < 0.05 and group >= "2020-03-01") or (score > 0.95 and group < "2020-03-01")
    else:
        raise NotImplementedError('The data type is not implemented yet.')


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
    filter_names = os.listdir(overlap_dir)    


    membership_info_path = os.path.join(save_dir, 'group_to_member.pkl')
    if not os.path.exists(membership_info_path) or not read_cache:
        # Process each file
        print("Going through each file to check BFF results...")
        membership_info = {}
        for i, (data_path, filename) in enumerate(tqdm(iterate_files(data_dir))):
            # # DEBUG:
            # if i > 3:
            #     break

            coverage_path = os.path.join(save_dir, '{}.pkl'.format(filename))
            if os.path.exists(coverage_path):
                with open(coverage_path, 'rb') as f:
                    total_coverages = pkl.load(f)
            else:
                # Load in the data
                data = custom_open(data_path)

                # is_member_all = [False] * len(data)
                total_coverages = []
                for filter_name in filter_names:
                    overlap_path = os.path.join(overlap_dir, filter_name, filename)
                    assert os.path.exists(overlap_path), overlap_path

                    # Read in each overlap file
                    overlap_data = custom_open(overlap_path)
                    assert len(data) == len(overlap_data)

                    total_coverages.append([calculate_coverage(dp) for dp in overlap_data])

                assert len(total_coverages) == len(filter_names)
                assert len(total_coverages[0]) == len(data)
                # aggreate the scores over the filters
                total_coverages = [max(sublist[i] for sublist in total_coverages) for i in range(len(total_coverages[0]))]
                total_coverages = {(filename, i): (total_coverages[i], get_group(data[i], data_type=data_type)) for i in range(len(total_coverages))}

                coverage_dir = os.path.dirname(coverage_path)
                if not os.path.exists(coverage_dir):
                    os.makedirs(coverage_dir)
                # Save the coverage information for the file
                with open((coverage_path), "wb") as f:
                    pkl.dump(total_coverages, f)

            # Build group dict and filter out unwanted ones
            for (filename, i), (score, group) in total_coverages.items():
                if group and qualified(data_type, score, group):
                    if group not in membership_info:
                        membership_info[group] = {'documents': []}    
                    membership_info[group]['documents'].append((filename, i, score))

        # # Save the membership info
        # with open(membership_info_path, "wb") as f:
        #     pkl.dump(membership_info, f)

    else:
        with open(membership_info_path, "rb") as f:
            membership_info = pkl.load(f)

    coverages_and_group = []
    for group, infos in membership_info.items():
        for (_, _, score) in infos['documents']:
            coverages_and_group.append((score, group))

    draw_separate_histogram(coverages_and_group, split=["1960", "2010", "2020-07-06", "2024"], xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=20)
        # draw_separate_histogram(total_coverage_member_values, xlabel="Percentage of duplication", ylabel="# Documents(k)",
        #                             save_path=os.path.join(save_dir, 'overlap_distribution2.png'), bins=20)

    # if data_type in ['rpj-arxiv']:
    #     # Create statistic info
    #     print("Calculating the statistics...")
    #     group_to_member = {group: members for group, members in group_to_member.items() if len(members) > 1}
    #     group_lengths = [len(members) for _, members in group_to_member.items()]
    #     group_member_rate = [sum([member[2] for member in members]) / len(members) for _, members in group_to_member.items()]
    #     sorted_group = sorted(group_to_member.items(), key=lambda x: x[0])
    #     sorted_group = [(group, len(members)) for group, members in sorted_group]
    #     stats = {
    #         "number of groups": len(group_to_member),
    #         "first n group": sorted_group[:15],
    #         "last n group": sorted_group[-15:],
    #         "number of group with all members": sum([rate == 1.0 for rate in group_member_rate]),
    #         "number of group with all non-members": sum([rate == 0.0 for rate in group_member_rate]),
    #         "average number of instance in every group": np.mean(group_lengths),
    #         "std number of instance in every group": np.std(group_lengths),
    #         "number of group with <= 1 documents": len([length for length in group_lengths if length <= 1]),
    #         "number of group with <= 3 documents": len([length for length in group_lengths if length <= 3]),
    #         "number of group with <= 5 documents": len([length for length in group_lengths if length <= 5]),
    #         "number of group with <= 10 documents": len([length for length in group_lengths if length <= 10]),
    #     }
    #     print(stats)
    #     with open(os.path.join(save_dir, 'stats.json'), "w") as f:
    #         json.dump(stats, f)
    #     draw_histogram(group_lengths, title=None, xlabel="# documents each date", ylabel="# Dates(k)",
    #                     save_path=os.path.join(save_dir, 'documents_date_distribution.png'), bins=50)
    #     draw_histogram(group_member_rate, title=None, xlabel="Percentage of Members", ylabel="# Dates(k)",
    #                     save_path=os.path.join(save_dir, 'memership_distribution.png'), bins=20, x_interval=0.05)
        
    #     # Create the check map
    #     check_map_path = os.path.join(save_dir, 'check_map.pkl')
    #     if not os.path.exists(check_map_path):
    #         check_map = {}
    #         for group, members in group_to_member.items():
    #             for (filename, i, is_member) in members:
    #                 check_map[filename, i] = (group, is_member)
    #         with open(check_map_path, "wb") as f:
    #             pkl.dump(check_map, f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/alrope/data/redpajama/arxiv/")
    parser.add_argument('--overlap_dir', type=str, default="/gscratch/h2lab/alrope/data/bff/redpajama-arxiv_newline_removed+pile")
    parser.add_argument('--save_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/redpajama-arxiv_newline_removed+pile_metas")
    parser.add_argument('--data_type', type=str, default="rpj-arxiv")
    parser.add_argument('--read_cache', action="store_true", default=False)

    args = parser.parse_args()

    main(args)