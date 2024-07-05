import os
import sys

import jsonlines
import numpy as np
import pudb
import torch
from transformers import AutoTokenizer

from datasets import Dataset


class Senti_Prompt_Data(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        args=None,
        emotions=["negative", "positive"],
        target=None,
    ):
        super(Senti_Prompt_Data, self).__init__()
        self.emotions = emotions
        self.tokenizer = tokenizer
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args
        self.target = target

        self.record = []
        self.read_content(json_path)

    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)

        with open(str(json_path), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                if self.target is not None:
                    target = self.target
                else:
                    target = np.random.randint(len(self.emotions))
                prompt = item["prompt"]["text"]

                context = self.tokenizer(prompt.strip(), return_tensors="np")[
                    "input_ids"
                ][0].tolist()

                if len(context) < 1:
                    continue

                target_label = self.tokenizer(
                    f"Sentiment: {self.emotions[target]}. ", return_tensors="np"
                )["input_ids"][0].tolist()
                self.record.append(
                    {
                        "query": torch.tensor(target_label + context, dtype=torch.long),
                        "prompt": torch.tensor(context, dtype=torch.long),
                        "target": target,
                    }
                )

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item


tokenizer = AutoTokenizer.from_pretrained("gpt2")

pu.db
neg_ds = Senti_Prompt_Data(
    os.environ.get("DATASETS_PATH") + "test/negative_prompts.jsonl", tokenizer
)
