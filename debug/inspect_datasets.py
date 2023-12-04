from datasets import load_dataset
import os

cache_dir = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/cache"
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, "transformers")

print(os.environ['HF_HOME'])
print(os.environ['HF_DATASETS_CACHE'])

dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train", cache_dir=os.path.join(cache_dir, "datasets"))


keys = set()

for dp in dataset:
    if dp["dataset"] not in keys:
        keys.add(dp["dataset"])
        print(dp["messages"][0])
        


print(keys)