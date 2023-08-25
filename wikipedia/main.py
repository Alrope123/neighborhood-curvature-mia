from collections import defaultdict, OrderedDict
import glob
import json
import mmap
import numpy as np
import os
import random
# from scipy.stats import entropy
import sys
import pickle as pkl
import time
import argparse
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

def main(args):
    save_dir = args.save_dir

    # Construct a tf.data.Dataset
    ds = tfds.load('wikipedia/20230601.ab', split='train', shuffle_files=False, data_dir=save_dir, download=True)

    # Build your input pipeline
    for example in ds.take(1):
        print(list(example.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="data")

    args = parser.parse_args()

    main(args)
