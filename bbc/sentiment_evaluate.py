import argparse
import os
import time
from datetime import timedelta

import torch
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
from evaluate import EvaluateConfig, evaluate
from reward_models import SentimentRewardModel
from train import prepare_ppo_trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from datasets import load_from_disk

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--base_models", nargs="+", default=["gpt2"], help="Base models")
parser.add_argument("--policy_model", type=str, default="gpt2", help="Policy model")
parser.add_argument("--description", type=str, default="", help="Run description")
args = parser.parse_args()
# Initialize evaluation variables
eval_config = EvaluateConfig()

# Set seed
seed = 0
torch.manual_seed(seed)
# Load policy model
print(f"Loading policy model {args.policy_model}")
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.policy_model)
# Load base models
base_models = []
base_model_tokenizers = []
for base_model_name in eval_config.base_models:
    base_models.append(AutoModelForCausalLM.from_pretrained(base_model_name))
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model_tokenizers.append(tokenizer)
# Load reward model
reward_model = SentimentRewardModel()

ppo_trainer = prepare_ppo_trainer(policy_model, None, eval_config)
ppo_trainer.accelerator.get_tracker("wandb").store_init_configuration(
    {"eval_config": eval_config}
)
if ppo_trainer.accelerator.is_main_process:
    print(eval_config)
    run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
    start = time.time()
else:
    run_id = None

run_id = broadcast_object_list([run_id])[0]
print(f"Process ID: {ppo_trainer.accelerator.process_index}: Run ID : {run_id}")
eval_config.run_id = run_id
test_file_names = [
    "positive_prompts_neg",
    "neutral_prompts_neg",
    "neutral_prompts_pos",
    "negative_prompts_pos",
]
if args.debug:
    test_datasets = []
    for file_name in test_file_names:
        debug_batch_size = 2
        ds = load_from_disk(
            os.environ.get("DATASETS_PATH") + "sentiment_prompts/" + file_name
        ).select(range(debug_batch_size * 2))
        test_datasets.append(ds)
    eval_config.project_name = "bbc-test"
    eval_config.batch_size = debug_batch_size
    eval_config.mini_batch_size = debug_batch_size
else:
    test_datasets = [
        load_from_disk(
            os.environ.get("DATASETS_PATH") + "sentiment_prompts/" + file_name
        )
        for file_name in test_file_names
    ]

logger = get_logger(__name__)
# Evaluate policy model
start = time.time()
evaluate(
    ppo_trainer,
    base_models,
    base_model_tokenizers,
    [reward_model],
    test_datasets,
    logger,
    eval_config,
)
if ppo_trainer.accelerator.is_main_process:
    end = time.time()
    print(f"Evaluation time: {timedelta(seconds=(end - start))}")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
