import os
import sys

import jsonlines
import numpy as np
import torch
from transformers import AutoTokenizer

from datasets import Dataset, DatasetInfo


class Senti_Prompt_Data(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        args=None,
        emotions=["negative", "positive"],
        target=None,
    ):
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
                        "target_label": self.emotions[target],
                    }
                )

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item


tokenizer = AutoTokenizer.from_pretrained("gpt2")

file_names = ["negative_prompts", "neutral_prompts", "positive_prompts"]
targets = [0, 1]
target_labels = ["neg", "pos"]
for file_name in file_names:
    for target in targets:
        ds = Senti_Prompt_Data(
            os.environ.get("DATASETS_PATH") + "test/" + file_name + ".jsonl",
            tokenizer,
            target=target,
        )
        ds_dict = {}
        ds_dict["query"] = [x["query"] for x in ds.record]
        ds_dict["target"] = [x["target"] for x in ds.record]
        ds_dict["prompt"] = [x["prompt"] for x in ds.record]
        ds_dict["target_label"] = [x["target_label"] for x in ds.record]

        ds_info = DatasetInfo(dataset_name=file_name + f"_{target_labels[target]}")

        dataset = Dataset.from_dict(ds_dict, info=ds_info)
        dataset.set_format("torch")
        dataset.save_to_disk(
            os.environ.get("DATASETS_PATH")
            + "sentiment_prompts/"
            + file_name
            + f"_{target_labels[target]}"
        )
