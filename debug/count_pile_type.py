import argparse
import json
import gzip
import json
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/sewon/data/the-pile/train-all-gz/Pile_CC-0.json.gz")

    args = parser.parse_args()

    types = []
    # Open the gzip file in read mode
    with gzip.open(args.data_path, 'rt', encoding='utf-8') as f:
        # Iterate through each line in the file
        for line in f:
            # Parse the JSON content from the line
            dp = json.loads(line)
            types.append(dp["meta"]["pile_set_name"])

    types = Counter(types)
    print(types)

# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
