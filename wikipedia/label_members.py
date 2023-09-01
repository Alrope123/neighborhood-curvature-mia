import os
import argparse

def iterate_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            yield file_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/alrope/data/wikipedia/processed/")
    parser.add_argument('--out_dir', type=str, default="out")

    args = parser.parse_args()

    for file_path in iterate_files(args.data_dir):
        print(file_path)
