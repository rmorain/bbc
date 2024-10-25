import glob
import os

import pandas as pd
from evaluate import margin_of_error

log_dir = "local_logs"
run_id = "3gu12mqc"
targets = ["neg", "pos"]
for target in targets:
    print(target)
    path = os.path.join(log_dir, run_id, f"*_{target}_*")

    all_files = glob.glob(path)
    df = pd.concat([pd.read_csv(f) for f in all_files]).Perplexity

    filtered_perplexity = df[df < 1e4]

    print(
        f"Total perplexity mean: {filtered_perplexity.mean()} +- {margin_of_error(filtered_perplexity)}"
    )
    print(f"Total perplexity median: {filtered_perplexity.median()}")
