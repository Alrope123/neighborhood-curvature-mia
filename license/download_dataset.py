from datasets import load_dataset
import json

def subsample_and_save(dataset_name, output_file):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    combined_data = []

    # Iterate through each subset
    for subset_name in dataset.keys():
        subset = dataset[subset_name]

        # Subsample 1000 rows
        subsample = subset.shuffle(seed=2024).select(range(1000))

        # Add the subset name as a field and append to combined_data
        for row in subsample:
            row_data = {**row, "subset_name": subset_name}
            combined_data.append(row_data)

    # Save to disk as .jsonl
    with open(output_file, 'w') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')

# Example usage
dataset_name = "kernelmachine/open-license-corpus"  # Replace with the dataset name
output_file = "/gscratch/h2lab/alrope/data/openlicense/0.jsonl"
subsample_and_save(dataset_name, output_file)
