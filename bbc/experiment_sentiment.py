import argparse
import os
from logging import Logger

import torch
from evaluate import EvaluateConfig, evaluate
from reward_models import SentimentRewardModel
from train import TrainingConfig, prepare_ppo_trainer, train
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
train_config = TrainingConfig()
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    train_config.model_name
)
base_model = AutoModelForCausalLM.from_pretrained(train_config.model_name)
base_model_tokenizers = [
    AutoTokenizer.from_pretrained(train_config.model_name, padding_side="left")
]
for t in base_model_tokenizers:
    t.pad_token = t.eos_token
train_dataset = load_from_disk(os.environ.get("DATASETS_PATH") + train_config.dataset)
if args.debug:
    debug_batch_size = 8
    train_dataset = train_dataset.select(range(debug_batch_size * 2))
    train_config.batch_size = debug_batch_size
    train_config.mini_batch_size = debug_batch_size
    train_config.project_name = "bbc-test"
reward_model = SentimentRewardModel()
logger = Logger(__name__)

ppo_trainer = prepare_ppo_trainer(policy_model, train_dataset, train_config)

# Train policy model
ppo_trainer = train(
    ppo_trainer,
    [base_model],
    base_model_tokenizers,
    [reward_model],
    logger,
    train_config,
)

# Save policy model
if not args.debug:
    # Create a directory for saved models if it doesn't exist
    save_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    run_id = ppo_trainer.accelerator.get_tracker("wandb").run._run_id
    model_dir = os.path.join(save_dir, f"{train_config.model_name}_{run_id}")
    policy_model.save_pretrained(model_dir)

# Initialize evaluation variables
eval_config = EvaluateConfig()

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
    eval_config.project_name = "bbc-test"
else:
    test_datasets = [
        load_from_disk(
            os.environ.get("DATASETS_PATH") + "sentiment_prompts/" + file_name
        )
        for file_name in test_file_names
    ]

# Evaluate policy model
evaluate(
    ppo_trainer,
    [base_model],
    base_model_tokenizers,
    [reward_model],
    test_datasets,
    logger,
    eval_config,
)
