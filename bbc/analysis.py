import os

import numpy as np
import pudb
from numpy import random

from datasets import concatenate_datasets, load_from_disk

debug = False
ds = load_from_disk(os.environ.get("DATASETS_PATH") + "imdb_sst2_tokenized_balanced")
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

# pos_examples = ds.filter(lambda x: x["target"] == 1)
# neg_examples = ds.filter(lambda x: x["target"] == 0).select(
#     list(range(len(pos_examples)))
# )

# pos_neg_examples = concatenate_datasets([pos_examples, neg_examples])

# if not debug:
#     pos_neg_examples.save_to_disk(
#         os.environ.get("DATASETS_PATH") + "imdb_sst2_tokenized_balanced"
#     )
