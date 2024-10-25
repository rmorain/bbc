import os

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset

df = pd.read_csv(
    "mmxbmk63.csv",
    names=["prefix", "prompt", "prompt_continuation", "correct", "reward"],
)
dataset = Dataset.from_pandas(df)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

base_models = [AutoModelForCausalLM.from_pretrained("gpt2-large").cuda()]
tokenizers = [AutoTokenizer.from_pretrained("gpt2")]
tokenizers[0].pad_token = tokenizers[0].eos_token

csv_file = os.path.join("mmxbmk63_perplexity.csv")


results = []
for batch in tqdm(dataloader, desc="Processing batches"):
    prompts = batch["prompt"]
    continuations = [
        prompt_cont[len(prompt) :]
        for prompt, prompt_cont in zip(batch["prompt"], batch["prompt_continuation"])
    ]
    p = perplexity(prompts, [continuations], base_models, tokenizers)
    results.extend(p.squeeze().tolist())
df = df[: len(results)]
df["perplexity"] = results
df = df[df["perplexity"] < 1e4]
df.to_csv("mmxbmk63_perplexity.csv")
print(f"Mean perplexity{df['perplexity'].mean()}")
print(f"Median perplexity{df['perplexity'].median()}")
