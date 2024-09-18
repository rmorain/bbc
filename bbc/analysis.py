import os

import numpy as np
from numpy import random

from datasets import load_from_disk

debug = False

dataset = "imdb_sst2_tokenized"
# dataset = "sst2_processed"

ds = load_from_disk(os.environ.get("DATASETS_PATH") + dataset)
if debug:
    indices = np.array(range(len(ds)))

    random.shuffle(indices)

    indices = indices[:8]

    ds = ds.select(indices)

pos_count = ds["target"].sum()
neg_count = len(ds) - pos_count

print(f"# positive examples: {pos_count} {pos_count / len(ds) * 100}%")
print(f"# negative examples: {neg_count} {neg_count / len(ds) * 100}%")
print(f"# Total examples: {len(ds)}")

# Analyse length
prompt_lengths = []
ds.map(lambda x: prompt_lengths.append(len(x["prompt"])))
print("Mean prompt length: ", np.mean(prompt_lengths))
print("STD of prompt length: ", np.std(prompt_lengths))
