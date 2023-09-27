import json
import argparse
import math
import os

def split_jsonl(filepath, total_lines, n):
    """
    Splits a JSONL file into n smaller JSONL files.

    :param filename: The name of the JSONL file.
    :type filename: str
    :param total_lines: The total number of lines in the JSONL file.
    :type total_lines: int
    :param n: The number of smaller files to split into.
    :type n: int
    """
    lines_per_file = math.ceil(total_lines / n)
    count = 0
    file_num = 1

    filedir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    with open(filepath, 'r') as file:
        outfile = open(os.path.join(filedir, f'{filename}_{file_num}.jsonl'), 'w')
        for line in file:
            if count < lines_per_file:
                outfile.write(line)
                count += 1
            else:
                outfile.close()
                file_num += 1
                outfile = open(os.path.join(filedir, f'{filename}_{file_num}.jsonl'), 'w')
                outfile.write(line)
                count = 1
        outfile.close()


# Command-line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a JSONL file into n smaller JSONL files.")
    parser.add_argument('--filename', type=str, help='The name of the JSONL file.')
    parser.add_argument('--total_lines', type=int, default=205744, help='The total number of lines in the JSONL file.')
    parser.add_argument('--n', type=int, default=100, help='The number of smaller files to split into.')

    args = parser.parse_args()

    split_jsonl(args.filename, args.total_lines, args.n)
