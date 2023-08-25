import argparse
import json
import io
import os
import zstandard as zstd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/sewon/data/the-pile/val.jsonl.zst")
    
    args = parser.parse_args()

    DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
    document_count = 0
    paragraph_count = 0
    samples = []
    with zstd.open(args.data_path, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
        for i, line in enumerate(iofh):
            document_count += 1
            paragraph_count += len(line.split('\n'))
            if i < 10:
                samples.append(line)
    print(document_count)
    print(paragraph_count)
    print("-------------")
    for sample in samples:
        print(sample)

# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
