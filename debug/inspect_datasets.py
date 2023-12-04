from datasets import load_dataset

dataset = load_dataset("allenai/tulu-v1-sft-mixture", split="train")

keys = set()

for dp in dataset:
    keys.add(dp["dataset"])

print(keys)