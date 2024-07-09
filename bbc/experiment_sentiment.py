import argparse
import os
from logging import Logger

import torch
from evaluate import EvaluateConfig, evaluate
from reward_models import SentimentRewardModel
from train import TrainingConfig, train
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from datasets import load_from_disk

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()
# Set seed
seed = 0
torch.manual_seed(seed)
# Initialize variables
config = TrainingConfig()
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
base_model_tokenizers = [
    AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
]
for t in base_model_tokenizers:
    t.pad_token = t.eos_token
train_dataset = load_from_disk(os.environ.get("DATASETS_PATH") + config.dataset)
if args.debug:
    debug_batch_size = 8
    train_dataset = train_dataset.select(range(debug_batch_size * 2))
    config.batch_size = debug_batch_size
    config.mini_batch_size = debug_batch_size
    config.project_name = "bbc-test"
reward_model = SentimentRewardModel()
logger = Logger(__name__)

policy_model = train(
    policy_model,
    [base_model],
    base_model_tokenizers,
    [reward_model],
    train_dataset,
    logger,
    config,
)

test_file_names = [
    "positive_prompts_neg",
    "neutral_prompts_neg",
    "neutral_prompts_pos",
    "negative_prompts_pos",
]
if args.debug:
    test_datasets = []
    for file_name in test_file_names:
        ds = load_from_disk(
            os.environ.get("DATASETS_PATH") + "sentiment_prompts/" + file_name
        ).select(range(2))
        test_datasets.append(ds)
    config.project_name = "bbc-test"
else:
    test_datasets = [
        load_from_disk(
            os.environ.get("DATASETS_PATH") + "sentiment_prompts/" + file_name
        )
        for file_name in test_file_names
    ]

evaluate(
    policy_model,
    [base_model],
    base_model_tokenizers,
    [reward_model],
    test_datasets,
)
