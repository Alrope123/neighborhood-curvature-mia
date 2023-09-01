import argparse
import json
import io
import os
import zstandard as zstd
import pickle as pkl
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/gscratch/h2lab/sewon/data/the-pile/train-all")
    parser.add_argument('--out_dir', type=str, default="out")

    args = parser.parse_args()

    assert os.path.exists(args.data_dir) and os.path.isdir(args.data_dir)
    for filename in os.listdir(args.data_dir):
        titles = set()
        file_path = os.path.join(args.data_dir, filename)

        DCTX = zstd.ZstdDecompressor(max_window_size=2**31)
        with zstd.open(file_path, mode='rb', dctx=DCTX) as zfh, io.TextIOWrapper(zfh) as iofh:
            for i, line in enumerate(iofh):
                dp = json.loads(line[:-1])
                if dp['meta']['pile_set_name'] == 'Wikipedia (en)':
                    text = dp['text']
                    titles.add(text.split('\n')[0])
        
        print("Colllected {} titiles for {}.".format(len(titles), filename))
        with open(os.path.join(args.out_dir, "{}.pkl".format(filename)), 'wb') as f:
            pkl.dump(titles, f)
    
    print("Combing all the sets")
    all_titles = set()
    for filename in tqdm(os.listdir(args.data_dir)):
        set_path = os.path.join(args.out_dir, "{}.pkl".format(filename))
        assert os.path.exists(set_path), "{} does not exists!".format(set_path)
        with open(set_path, 'rb') as f:
            cur_set = pkl.load(f)
            all_titles.update(cur_set)
    with open(os.path.join(args.out_dir, "all.pkl"), 'wb') as f:
        pkl.dump(all_titles, f)

