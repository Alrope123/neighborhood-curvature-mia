import json
import numpy as np
import os
# from scipy.stats import entropy
import pickle as pkl
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    cover_length = sum([x[1] - x[0] for x in dp["bff_duplicate_spans"]])
    total_length = dp["length"]
    return cover_length / total_length if total_length > 0  else 0

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
        if nonmember_dict == {}:
            with open(nonmember_dict_path, 'rb') as f:
                nonmember_dict = pkl.load(f)
        title = dp['title']
        if title in member_dict and member_dict[title] != None:
            return member_dict[title].split(',')[1].strip()
        elif title in nonmember_dict and nonmember_dict[title] != None:
            return nonmember_dict[title].split(',')[1].strip()
        else:
            return None
    else:
        raise NotImplementedError('The data type is not implemented yet.')


def draw_histogram(data, save_path, bins=None, title=None, xlabel=None, ylabel=None, cumulative=False, x_interval=-1):
    """Draw a histogram for the given data."""
    
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


def main(args):
    # Process args
    data_dir = args.data_dir
    overlap_dir = args.overlap_dir
    save_dir = args.save_dir
    data_type = args.data_type
    threshold = args.threshold

    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, str(threshold))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filter_names = os.listdir(overlap_dir)    


    membership_info_path = os.path.join(save_dir, 'group_to_member.pkl')
    # drawn = {filter_name : False for filter_name in filter_names}
    total_coverages = []
    if not os.path.exists(membership_info_path):
        # Process each file
        print("Going through each file to check BFF results...")
        group_to_member = {}
        for i, (data_path, filename) in enumerate(tqdm(iterate_files(data_dir))):
            # DEBUG:
            # if i > 10:
            #     break

            # # Figure out the path
            # data_path = os.path.join(data_dir, filename)
            

            # Load in the data
            data = custom_open(data_path)

            is_member_all = [False] * len(data)

            for filter_name in filter_names:
                overlap_path = os.path.join(overlap_dir, filter_name, filename)
                assert os.path.exists(overlap_path), overlap_path

                # Read in each overlap file
                overlap_data = custom_open(overlap_path)
                assert len(data) == len(overlap_data)
                coverages = [calculate_coverage(dp) for dp in overlap_data]
                total_coverages.extend(coverages)
                # Draw the distribution of overlaps if haven't drawn
                # if not drawn[filter_name]:
                #     draw_histogram(coverages, title=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                #                     save_path=os.path.join(save_dir, 'overlap_distribution_{}.png'.format(filter_name)), bins=50, x_interval=0.01)
                #     draw_histogram(coverages, title=None, xlabel="Percentage of duplication",
                #                     save_path=os.path.join(save_dir, 'overlap_distribution_CDF_{}.png'.format(filter_name)), bins=50, cumulative=True, x_interval=0.02)
                #     drawn[filter_name] = True
                is_member = [coverage > threshold for coverage in coverages]
                assert len(is_member_all) ==  len(is_member)
                is_member_all = [a or b for a, b in zip(is_member_all, is_member)]

            cur_save_dir = os.path.dirname(os.path.join(save_dir, '{}.pkl'.format(filename)))
            if not os.path.exists(cur_save_dir):
                os.makedirs(cur_save_dir)
            with open(os.path.join(save_dir, '{}.pkl'.format(filename)), "wb") as f:
                pkl.dump(is_member_all, f)

            for i, dp in enumerate(data):
                group = get_group(dp, data_type)
                if group:
                    if group not in group_to_member:
                        group_to_member[group] = []
                    group_to_member[group].append((filename, i, is_member_all[i]))

        # Save the membership info
        with open(membership_info_path, "wb") as f:
            pkl.dump(group_to_member, f)
    
    else:
        with open(membership_info_path, "rb") as f:
            group_to_member = pkl.load(f)

    draw_histogram(total_coverages, title=None, xlabel="Percentage of duplication", ylabel="# Documents(k)",
                                    save_path=os.path.join(save_dir, 'overlap_distribution.png'), bins=50, x_interval=0.02)
    draw_histogram(total_coverages, title=None, xlabel="Percentage of duplication",
                    save_path=os.path.join(save_dir, 'overlap_distribution_CDF.png'), bins=50, cumulative=True, x_interval=0.02)

    # Create statistic info
    print("Calculating the statistics...")
    group_to_member = {group: members for group, members in group_to_member.items() if len(members) > 1}
    group_lengths = [len(members) for _, members in group_to_member.items()]
    group_member_rate = [sum([member[2] for member in members]) / len(members) for _, members in group_to_member.items()]
    sorted_group = sorted(group_to_member.items(), key=lambda x: x[0])
    sorted_group = [(group, len(members)) for group, members in sorted_group]
    stats = {
        "threshold": threshold,
        "number of groups": len(group_to_member),
        "first n group": sorted_group[:15],
        "last n group": sorted_group[-15:],
        "number of group with all members": sum([rate == 1.0 for rate in group_member_rate]),
        "number of group with all non-members": sum([rate == 0.0 for rate in group_member_rate]),
        "average number of instance in every group": np.mean(group_lengths),
        "std number of instance in every group": np.std(group_lengths),
        "number of group with <= 1 documents": len([length for length in group_lengths if length <= 1]),
        "number of group with <= 3 documents": len([length for length in group_lengths if length <= 3]),
        "number of group with <= 5 documents": len([length for length in group_lengths if length <= 5]),
        "number of group with <= 10 documents": len([length for length in group_lengths if length <= 10]),
    }
    print(stats)
    with open(os.path.join(save_dir, 'stats.json'), "w") as f:
        json.dump(stats, f)
    draw_histogram(group_lengths, title=None, xlabel="# documents each date", ylabel="# Dates(k)",
                    save_path=os.path.join(save_dir, 'documents_date_distribution.png'), bins=50)
    draw_histogram(group_member_rate, title=None, xlabel="Percentage of Members", ylabel="# Dates(k)",
                    save_path=os.path.join(save_dir, 'memership_distribution.png'), bins=20, x_interval=0.05)
    
    # Create the check map
    check_map_path = os.path.join(save_dir, 'check_map.pkl')
    if not os.path.exists(check_map_path):
        check_map = {}
        for group, members in group_to_member.items():
            for (filename, i, is_member) in members:
                check_map[filename, i] = (group, is_member)
        with open(check_map_path, "wb") as f:
            pkl.dump(check_map, f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/alrope/data/redpajama/arxiv/")
    parser.add_argument('--overlap_dir', type=str, default="/gscratch/h2lab/alrope/data/bff/redpajama-arxiv+pile")
    parser.add_argument('--save_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv")
    parser.add_argument('--data_type', type=str, default="rpj-arxiv")
    parser.add_argument('--threshold', type=float, default="0.1")

    args = parser.parse_args()

    main(args)
