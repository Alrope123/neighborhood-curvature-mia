import os
import argparse
import pickle as pkl
import json
from tqdm import tqdm

def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_early_dir', type=str, default="/gscratch/h2lab/alrope/data/wikipedia/processed3/")
    parser.add_argument('--data_late_dir', type=str, default="/gscratch/h2lab/alrope/data/wikipedia/processed2/")
    parser.add_argument('--out_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out2")

    args = parser.parse_args()

    pile_set = set()

    # initialize the member/non-member sets
    nonmember_text_path = os.path.join(args.out_dir, 'pile_nonmember_text.pkl')
    # if os.path.exists(nonmember_text_path):
    #     with open(nonmember_text_path, 'rb') as f:
    #         nonmember_text_set = pkl.load(f)  
    # else:
    # Go through early set
    early_set = set()
    for file_path, filename in tqdm(iterate_files(args.data_early_dir)):
        with open(file_path, 'r') as f:
            for line in f:
                dp = json.loads(line)
                title = dp['title']
                if len(dp['text'].split()) > 0:
                    early_set.add(title)
    # Go through late set
    late_set = set()
    for file_path, filename in tqdm(iterate_files(args.data_late_dir)):
        with open(file_path, 'r') as f:
            for line in f:
                dp = json.loads(line)
                title = dp['title']
                if len(dp['text'].split()) > 0:
                    late_set.add(title)

    nonmember_text_set = late_set.difference(early_set)

    # Save
    with open(nonmember_text_path, 'wb') as f:
        pkl.dump(nonmember_text_set, f)


    print("# articles from newest Wikipedia dump in the pile with text available: {}".format(len(nonmember_text_set)))

