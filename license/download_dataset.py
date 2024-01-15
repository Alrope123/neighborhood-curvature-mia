from datasets import load_dataset
import json

def subsample_and_save(dataset_name, output_file):
    # Load the dataset
    subset_names = ['ccby_law', 'ccby_s2orc', 'ccby_stackexchange', 'ccby_stackoverflow', 'ccby_wikinews', 'ccby_wikipedia', 'pd_arxiv_abstracts', 'pd_books', 'pd_law', 'pd_news', 'pd_s2orc', 'sw_amps_math', 'sw_dm_math', 'sw_github', 'sw_hackernews', 'sw_ubuntu_irc']
    combined_data = []

    # Iterate through each subset
    for subset_name in subset_names:
        subset = load_dataset('kernelmachine/open-license-corpus', 'pd_law', streaming=True)['train']

        # Subsample 1000 rows
        subsample = subset.shuffle(seed=2024)[:1000]

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
