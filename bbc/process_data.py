import os

import pudb
from transformers import AutoTokenizer
from trl.core import LengthSampler

from datasets import load_from_disk


# TODO
def process_data(dataset_name, tokenizer, input_min_text_length, input_max_text_length):
    ds = load_from_disk(os.environ.get("DATASETS_PATH") + dataset_name)
    pu.db

    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    ds = ds.map(
        tokenize,
        batched=False,
        fn_kwargs={"input_size": input_size, "tokenizer": tokenizer},
    )
    ds = ds.remove_columns(["review", "label"])

    ds.set_format(type="torch")
    return ds


def tokenize(sample, input_size=None, tokenizer=None):
    if sample["label"] == 0:
        sample["target"] = 1
    else:
        sample["target"] = 0
    sample["target_label"] = emotions[sample["target"]]
    input_size = input_size()
    sample["prompt"] = tokenizer.encode(sample["review"])[:input_size]
    sample["query"] = tokenizer.encode(
        f"Sentiment: {emotions[sample['target']]}. {tokenizer.decode(sample['prompt'])}"
    )
    return sample


if __name__ == "__main__":
    dataset_name = "imdb_sst2"
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = process_data(dataset_name, tokenizer)
