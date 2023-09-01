import argparse
import json
import io
import os
import zstandard as zstd
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/sewon/data/the-pile/val.jsonl.zst")
    parser.add_argument('--out_dir', type=str, default=".")

    args = parser.parse_args()

    DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
    lines = []
    with zstd.open(args.data_path, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
        for i, line in enumerate(iofh):
            if i % 1000 == 0:
                lines.append(json.loads(line[:-1]))
    
    with open(os.path.join(args.out_dir, "sample.jsonl"), 'w') as f:
        for line in lines:
            f.write(json.dumps(line))
            f.write('\n')
    

# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
