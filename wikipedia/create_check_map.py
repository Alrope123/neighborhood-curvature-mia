import os
import argparse
import pickle as pkl
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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


def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file)
    return file_paths, file_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/alrope/data/wikipedia/processed/")
    parser.add_argument('--member_dict_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/pile_member_text_w_time.pkl")
    parser.add_argument('--nonmember_dict_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/pile_nonmember_text_w_time.pkl")
    parser.add_argument('--out_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out")

    args = parser.parse_args()

    # initialize the check map
    check_map_path = os.path.join(args.out_dir, 'check_map.pkl')
    if os.path.exists(check_map_path):
        with open(check_map_path, 'rb') as f:
            check_map = pkl.load(f)
    else:
        check_map = {}
        # Load in sets with the time
        assert os.path.exists(args.member_dict_path), args.member_dict_path
        assert os.path.exists(args.nonmember_dict_path), args.member_dict_path
        with open(args.member_dict_path, 'rb') as f:
            member_dict = pkl.load(f)
        with open(args.nonmember_dict_path, 'rb') as f:
            nonmember_dict = pkl.load(f)

        # Iterate each file
        file_paths, filenames = iterate_files(args.data_dir)
        for file_path, filename in tqdm(zip(file_paths, filenames)):
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    dp = json.loads(line)
                    title = dp['title']
                    if title in member_dict and member_dict[title] != None:
                        date = member_dict[title].split(',')[1].strip()
                        check_map[filename, i] = (date, True)
                    elif title in nonmember_dict and nonmember_dict[title] != None:
                        date = nonmember_dict[title].split(',')[1].strip()
                        check_map[filename, i] = (date, False)
        
        # Save the checkmap
        with open(check_map_path, 'wb') as f:
            pkl.dump(check_map, f)


    # Build group to members
    group_to_members = {}
    for _, (date, is_member) in check_map.items():
        if date not in group_to_members:
            group_to_members[date] = []
        group_to_members[date].append(is_member)

    group_lengths = [len(members) for _, members in group_to_members.items()]
    group_member_rate = [sum(members) / len(members) for _, members in group_to_members.items()]
    stats = {
        "# groups": len(group_to_members),
        "number of group with all members": sum([all(members) for _, members in group_to_members]),
        "number of group with all non-members": sum([not any(members) for _, members in group_to_members]),
        "average number of instance in every group": np.mean(group_lengths),
        "std number of instance in every group": np.std(group_lengths), 
    }
    draw_histogram(group_member_rate, title=None, xlabel="Percentage of Members", ylabel="# Dates(k)",
                    save_path=os.path.join(args.save_dir, 'memership_distribution.png'), bins=20, x_interval=0.05)
    print(stats)
    with open(os.path.join(args.out_dir, "stats.json"), 'w') as f:
        json.dump(check_map, f)
