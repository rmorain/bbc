import numpy as np
import pandas as pd
from train import distinctness

combined_files = pd.read_csv(
    "local_logs/ckdia6a0/positive_prompts_neg_log_ckdia6a0.csv"
)
unigram = []
bigram = []
trigram = []

num_chunks = len(combined_files)
for chunk in np.array_split(combined_files["Continuation"], num_chunks):
    u, b, t = distinctness(chunk)
    unigram.append(u)
    bigram.append(b)
    trigram.append(t)
print("done")
