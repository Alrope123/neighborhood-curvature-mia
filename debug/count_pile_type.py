import argparse
import json
import gzip
import json
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/data/data/pile/val.jsonl")

    args = parser.parse_args()

    types = []
    # Open the gzip file in read mode
    with open(args.data_path, 'r') as f:
        # Iterate through each line in the file
        for line in f:
            # Parse the JSON content from the line
            dp = json.loads(line)
            types.append(dp["meta"]["pile_set_name"])

    types = Counter(types)
    print(types)

# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
