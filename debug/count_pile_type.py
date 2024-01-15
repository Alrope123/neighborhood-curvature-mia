import argparse
import json
import io
import gzip
import json

def tokenize(s):
    words = re.findall(r'\S+|\s+', s)
    return list(filter(lambda w: not w.isspace(), words))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/sewon/data/the-pile/train-all-gz/Pile_CC-0.json.gz")

    args = parser.parse_args()

    data = []
    # Open the gzip file in read mode
    with gzip.open(args.data_path, 'rt', encoding='utf-8') as f:
        # Iterate through each line in the file
        for line in f:
            # Parse the JSON content from the line
            dp = json.loads(line)
            data.append(dp)
            print(dp)



# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
