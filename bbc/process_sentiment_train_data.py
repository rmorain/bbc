import os

import pudb
from transformers import AutoTokenizer
from trl.core import LengthSampler

from datasets import concatenate_datasets, load_from_disk


def process_sentiment_data(
    dataset_name,
    tokenizer,
    emotions=["negative", "positive"],
    input_min_text_length=5,
    input_max_text_length=13,
    balance=True,
):
    ds = load_from_disk(os.environ.get("DATASETS_PATH") + dataset_name)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    ds = ds.map(
        tokenize,
        batched=False,
        fn_kwargs={
            "input_size": input_size,
            "tokenizer": tokenizer,
            "emotions": emotions,
        },
    )
    ds = ds.remove_columns(["text", "label"])

    ds.set_format(type="torch")
    # Balance dataset
    if balance:
        pos_examples = ds.filter(lambda x: x["target_label"] == "positive")
        neg_examples = ds.filter(lambda x: x["target_label"] == "negative")
        limit = (
            len(pos_examples)
            if len(pos_examples) < len(neg_examples)
            else len(neg_examples)
        )
        pos_examples = pos_examples.select(range(limit))
        neg_examples = neg_examples.select(range(limit))
        ds = concatenate_datasets([neg_examples, pos_examples])
    return ds


def tokenize(sample, input_size=None, tokenizer=None, emotions=None):
    if sample["label"] == 0:
        sample["target"] = 1
    else:
        sample["target"] = 0
    sample["target_label"] = emotions[sample["target"]]
    input_size = input_size()
    sample["prompt"] = tokenizer.encode(sample["text"])[:input_size]
    sample["query"] = tokenizer.encode(
        f"Sentiment: {emotions[sample['target']]}. {tokenizer.decode(sample['prompt'])}"
    )
    return sample


if __name__ == "__main__":
    dataset_name = "imdb_sst2"
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = process_sentiment_data(dataset_name, tokenizer)
    ds.save_to_disk(os.environ.get("DATASETS_PATH") + "imdb_sst2_processed")
