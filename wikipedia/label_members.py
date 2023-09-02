import os
import argparse
import pickle as pkl
import json
import tqdm as tqdm

def iterate_files(root_dir):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/alrope/data/wikipedia/processed/")
    parser.add_argument('--pile_set_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/all_pile_set.pkl")
    parser.add_argument('--out_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out")

    args = parser.parse_args()

    # Load in the set that defines member
    with open(args.pile_set_path, 'rb') as f:
        pile_set = pkl.load(f)

    # initialize the member/non-member sets
    member_path = os.path.join(args.out_dir, 'pile_member.pkl')
    non_member_path = os.path.join(args.out_dir, 'pile_nonmember.pkl')
    if os.path.exists(member_path) and os.path.exists(non_member_path):
        with open(member_path, 'rb') as f:
            member_set = pkl.load(f)
        with open(non_member_path, 'rb') as f:
            non_member_set = pkl.load(f)  
    else:
        member_set = set()
        non_member_set = set()
        # Go through each file and label them based on title
        for file_path in tqdm(iterate_files(args.data_dir)):
            with open(file_path, 'r') as f:
                for line in f:
                    dp = json.loads(line)
                    if dp['text'] != "":
                        title = dp['title']
                        if title in pile_set:
                            member_set.add(title)
                        else:
                            non_member_set.add(title)
        with open(member_path, 'wb') as f:
            pkl.dump(member_set, f)

        with open(non_member_path, 'wb') as f:
            pkl.dump(non_member_set, f)

    print("Total # articles in the pile: {}".format(len(pile_set)))
    print("# articles from newest Wikipedia dump in the pile: {}".format(len(member_set)))
    print("# articles from newest Wikipedia dump NOT in the pile: {}".format(len(non_member_set)))

