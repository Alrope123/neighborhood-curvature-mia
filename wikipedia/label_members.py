import os
import argparse
import pickle as pkl
import json
from tqdm import tqdm

def camel_case_split(str):
    words = [[str[0]]]
 
    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
 
    return ' '.join([''.join(word) for word in words])
     

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
    member_text_path = os.path.join(args.out_dir, 'pile_member_text.pkl')
    non_member_text_path = os.path.join(args.out_dir, 'pile_nonmember_text.pkl')
    member_notext_path = os.path.join(args.out_dir, 'pile_member_text.pkl')
    non_member_notext_path = os.path.join(args.out_dir, 'pile_nonmember_text.pkl')
    if os.path.exists(member_text_path) and os.path.exists(non_member_text_path) and \
        os.path.exists(member_notext_path) and os.path.exists(non_member_notext_path):
        with open(member_text_path, 'rb') as f:
            member_text_set = pkl.load(f)
        with open(non_member_text_path, 'rb') as f:
            non_member_text_set = pkl.load(f)
        with open(member_notext_path, 'rb') as f:
            member_notext_set = pkl.load(f)
        with open(non_member_notext_path, 'rb') as f:
            non_member_notext_set = pkl.load(f)  
    else:
        member_text_set = set()
        non_member_text_set = set()
        member_notext_set = set()
        non_member_notext_set = set()
        # Go through each file and label them based on title
        for file_path in tqdm(iterate_files(args.data_dir)):
            with open(file_path, 'r') as f:
                for line in f:
                    dp = json.loads(line)
                    title = dp['title']
                    if title in pile_set # or camel_case_split(title) in pile_set:
                        if dp['text'] != "":
                            member_text_set.add(title)
                        else:
                            member_notext_set.add(title)
                    else:
                        if dp['text'] != "":
                            non_member_text_set.add(title)
                        else:
                            non_member_notext_set.add(title)
        # Save
        with open(member_text_path, 'wb') as f:
            pkl.dump(member_text_set, f)
        with open(non_member_text_path, 'wb') as f:
            pkl.dump(non_member_text_set, f)
        with open(member_notext_path, 'wb') as f:
            pkl.dump(member_notext_set, f)
        with open(non_member_notext_path, 'wb') as f:
            pkl.dump(non_member_notext_set, f)

    print("Total # articles in the pile: {}".format(len(pile_set)))
    print("# articles from newest Wikipedia dump in the pile with text available: {}".format(len(member_text_set)))
    print("# articles from newest Wikipedia dump NOT in the pile: with text available: {}".format(len(non_member_text_set)))
    print("# articles from newest Wikipedia dump in the pile without text available: {}".format(len(member_notext_set)))
    print("# articles from newest Wikipedia dump NOT in the pile: without text available: {}".format(len(non_member_notext_set)))

