import argparse
import json
import io
import os
import zstandard as zstd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/sewon/data/the-pile/val.jsonl.zst")
    parser.add_argument('--check_id', action="store_true", default=False)

    args = parser.parse_args()

    DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
    document_count = 0
    # paragraph_count = 0
    samples = []
    unmatched = []
    j = 0
    with zstd.open(args.data_path, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
        for i, line in enumerate(iofh):
            document_count += 1
            print(line[:-1])
            dp = json.loads(line[:-1])
            print("!")
            assert False
            # paragraph_count += len(dp['text'])
            if args.check_id:
                assert j <= dp['id']
                while j != dp['id']:
                    unmatched.append(j)
                    j += 1
                j += 1
                assert j == dp['id'] + 1
            if i < 10:
                samples.append(line)
    print(document_count)
    # print(paragraph_count)
    print(len(unmatched))
    print(unmatched)
    print("-------------")
    for sample in samples:
        print(sample)

# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
