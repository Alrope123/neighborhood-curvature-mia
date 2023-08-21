import argparse
import json
import io
import os

# REDPAJAMA_DOMAINS = ["arxiv", "book", "c4", "cc_2019-30", "cc_2020-05", "cc_2021-04", "cc_2022-05", "cc_2023-06", "github", "stackexchange", "wikipedia"]
REDPAJAMA_DOMAINS = ["arxiv", "book", "github", "stackexchange", "wikipedia", "cc_2019-30"]
PILE_DOMAINS = ["PubMed Abstracts", "PubMed Central", "Github", "StackExchange", "Enron Emails", "FreeLaw", "USPTO Backgrounds", "Wikipedia (en)", "Books3", "HackerNews", "Gutenberg (PG-19)", "DM Mathematics", "NIH ExPorter", "ArXiv", "BookCorpus2", "OpenSubtitles", "YoutubeSubtitles", "Ubuntu IRC", "EuroParl", "PhilPapers", "Pile-CC"]
# REDPAJAMA_DISTRIBUTION = [28, 26, 175, 878, 59, 20, 24]
# REDPAJAMA_DISTRIBUTION = [28, 26, 59, 20, 24]

cache_dir = "cache"
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, "transformers")


def main(args):
    samples = {}
    
    if args.dataset == "redpajama":
        for domain in REDPAJAMA_DOMAINS:
            assert domain in REDPAJAMA_DOMAINS
            with open(os.path.join(args.data_dir, f"{domain}_sample.jsonl"), 'r') as f:
                for line in f:
                    samples[domain] = json.loads(line)
                    break

    elif args.dataset == "pile":
        import zstandard as zstd
        DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
        with zstd.open(args.dataset, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
            for i, line in enumerate(iofh):
                if i < 1000:
                    data = json.loads(line) 
                    samples[data['meta']['pile_set_name']] = data
                else:
                    break
    
    with open('{}_sample.json'.format(args.dataset), 'w') as f:
        json.dump(samples, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with multiple arguments")
    parser.add_argument("--data_dir", type=str, default="cache/datasets/datasets--togethercomputer--RedPajama-Data-1T-Sample/snapshots/98f93b765c118b999b1af570c3e399c207d08da7")
    parser.add_argument("--dataset", type=str, default="redpajama")
    # parser.add_argument("--domains", nargs="+", required=True)

    args = parser.parse_args()

    main(args)

# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
