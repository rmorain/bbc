import argparse
import os

import torch
from accelerate.logging import get_logger
from evaluate import EvaluateConfig, evaluate
from reward_models import SentimentRewardModel
from train import TrainingConfig, prepare_ppo_trainer, train
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from datasets import load_from_disk

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--base_models", nargs="+", default=["gpt2"], help="Base models")
parser.add_argument("--policy_model", type=str, default="gpt2", help="Policy model")
parser.add_argument("--description", type=str, default="", help="Run description")

args = parser.parse_args()
# Set seed
seed = 0
torch.manual_seed(seed)
# Initialize variables
train_config = TrainingConfig(
    num_epochs=args.num_epochs,
    policy_model=args.policy_model,
    base_models=args.base_models,
    tracker_kwargs={"wandb": {"notes": args.description}},
)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    train_config.policy_model
)
base_models = []
base_model_tokenizers = []
for base_model_name in train_config.base_models:
    base_models.append(AutoModelForCausalLM.from_pretrained(base_model_name))
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model_tokenizers.append(tokenizer)

train_dataset = load_from_disk(os.environ.get("DATASETS_PATH") + train_config.dataset)
if args.debug:
    debug_batch_size = 8
    train_dataset = train_dataset.select(range(debug_batch_size * 2))
    train_config.batch_size = debug_batch_size
    train_config.mini_batch_size = debug_batch_size
    train_config.project_name = "bbc-test"
reward_model = SentimentRewardModel()

ppo_trainer = prepare_ppo_trainer(policy_model, train_dataset, train_config)
ppo_trainer.accelerator.get_tracker("wandb").store_init_configuration(
    {"train_config": train_config}
)

logger = get_logger(__name__)
# Train policy model
ppo_trainer = train(
    ppo_trainer,
    base_models,
    base_model_tokenizers,
    [reward_model],
    logger,
    train_config,
)

# Save policy model
if not args.debug and ppo_trainer.accelerator.is_main_process:
    # Create a directory for saved models if it doesn't exist
    save_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
    model_dir = os.path.join(save_dir, f"{train_config.policy_model}_{run_id}")
    policy_model.save_pretrained(model_dir)
    print(f"Policy model saved at {model_dir}")

# Initialize evaluation variables
eval_config = EvaluateConfig()

ppo_trainer.accelerator.get_tracker("wandb").store_init_configuration(
    {"eval_config": eval_config}
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
    base_models,
    base_model_tokenizers,
    [reward_model],
    test_datasets,
    logger,
    eval_config,
)
