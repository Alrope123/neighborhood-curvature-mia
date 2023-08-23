import argparse
import json
import io
import os


if __name__ == "__main__":
    import zstandard as zstd
    DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
    document_count = 0
    paragraph_count = 0
    with zstd.open("/gscratch/h2lab/sewon/data/the-pile/val.jsonl.zst", mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
        for i, line in enumerate(iofh):
            document_count += 1
            paragraph_count += len(line.split('\n'))
    print(document_count)
    print(paragraph_count)

# python generate_predctions.py --model_name togethercomputer/RedPajama-INCITE-Base-3B-v1 --n 1000 --answer_n 3
