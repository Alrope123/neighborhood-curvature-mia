import argparse
import json
import io
import os
import zstandard as zstd
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/gscratch/h2lab/sewon/data/the-pile/val.jsonl.zst")
    
    args = parser.parse_args()

    DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
    with zstd.open(args.data_path, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
        for i, line in enumerate(iofh):
            dp = json.loads(line[:-1])
            if dp['meta']['pile_set_name'] == 'Wikipedia (en)':
                text = dp['text']
                title = text.split('\n')[:2]
                print(title)
                assert False
