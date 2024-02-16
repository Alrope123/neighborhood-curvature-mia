import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/alrope/data/redpajama/arxiv/")
    parser.add_argument('--key', type=str, default=None)
    args = parser.parse_args()

    short_book_titles = set()
    titles = set()
    with open(args.data_path, 'r') as f:
        # Load the JSONL file
        if args.data_path.endswith(".jsonl"):    
            for i, line in enumerate(f):
                dp = json.loads(line)
                if 'title' in dp['meta']:
                    titles.add(dp['meta']['title'])
                else:
                    short_book_titles.add(dp['meta']['short_book_title'])
                if i > 20:
                    break
        else:
            data = json.load(f)

    print(short_book_titles)
    print(titles)

