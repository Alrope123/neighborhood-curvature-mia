import argparse
import json
import io
import os
import zstandard as zstd
import re

def tokenize(s):
    words = re.findall(r'\S+|\s+', s)
    return list(filter(lambda w: not w.isspace(), words))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/sewon/data/the-pile/val.jsonl.zst")
    parser.add_argument('--check_id', action="store_true", default=False)
    parser.add_argument('--n_gram', type=int, default=13)

    args = parser.parse_args()

    DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
    document_count = 0
    # paragraph_count = 0
    n_gram_count = 0
    samples = []
    unmatched = []
    j = 0
    with zstd.open(args.data_path, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
        for i, line in enumerate(iofh):
            document_count += 1        
            if args.check_id:
                # print(line[:-1])
                try:
                    dp = json.loads(line[:-1])
                except ValueError as e:
                    print(j)
                    print([line[:-1]])
                    j += 1
                    continue
                assert j <= dp['id']
                while j != dp['id']:
                    unmatched.append(j)
                    j += 1
                j += 1
                assert j == dp['id'] + 1
            dp = json.loads(line[:-1])
            n_gram_count += len(tokenize(dp['text']))-args.n_gram+1
            if i < 10:
                samples.append(line)
    print(document_count)
    # print(paragraph_count)
    print(len(unmatched))
    print(unmatched)
    print("-------------")
    for sample in samples:
        print(sample)
    print(n_gram_count)

# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
