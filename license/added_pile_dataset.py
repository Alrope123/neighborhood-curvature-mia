from datasets import load_dataset
import json
import random
import argparse

def subsample_and_save(dataset_path, output_file, subsample_size=1000):
    random.seed(2024)

    subset_names = ['Books3', 'OpenWebText2', 'BookCorpus2', 'Enron Emails']
    subset_data = {name: [] for name in subset_names}
    complete = {name: False for name in subset_names}

    # Open the gzip file in read mode
    with open(dataset_path, 'r') as f:
        # Iterate through each line in the file
        for line in f:
            # Parse the JSON content from the line
            dp = json.loads(line)
            t = dp["meta"]["pile_set_name"]
            if t in subset_data and not complete[t]:
                subset_data[t].append(dp["text"])
                if len(subset_data[t]) > subsample_size:
                    complete[t] = True
                    print("!!!!!!!!!!!!!!!!!!!")
                if all([c for name, c in complete.items()]):
                    assert False
                    break

    print({name: len(text) for name, text in subset_data.items()})

    with open(output_file, 'w') as f:
        for name, text_array in subset_data.items():
            for text in text_array:
                f.write(json.dumps({"subset_name": name, "text": text}) + '\n')


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="/data/data/pile/val.jsonl")
args = parser.parse_args()

# Example usage
dataset_path = args.data_path  # Replace with the dataset name
output_file = "/gscratch/h2lab/alrope/data/openlicense/1.jsonl"
subsample_and_save(dataset_path, output_file)
