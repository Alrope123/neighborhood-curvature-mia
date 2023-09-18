import json
import argparse

def print_structure(data, indent=0, key=''):
    """Recursively prints the structure of a JSON object."""
    # Base case: if the current data is a list
    if isinstance(data, list):
        if data:
            print(' ' * indent + key + ": List of")
            print_structure(data[0], indent + 2, "item")
        else:
            print(' ' * indent + key + ": Empty List")

    # Base case: if the current data is a dictionary
    elif isinstance(data, dict):
        print(' ' * indent + key + ":")
        for key, value in data.items():
            print_structure(value, indent + 2, key)

    # Base case: for other data types
    else:
        print(' ' * indent + key + f": {type(data).__name__}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/alrope/data/redpajama/arxiv/")
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        # Load the JSONL file
        if args.data_path.endswith(".jsonl"):    
            for line in f:
                data = json.load(line)
                break
        else:
            data = json.load(f)
            
    print_structure(data)
