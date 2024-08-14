import argparse
import csv
import os
import time
import traceback
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List

import numpy as np
import psutil
import scipy.stats as stats
import torch
from accelerate.utils import broadcast_object_list
from reward_models import RewardModel, SentimentRewardModel
from scipy.stats import binomtest
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from train import (
    TrainingConfig,
    compute_reward,
    distinctness,
    generate_prefix,
    local_log,
    perplexity,
    prepare_ppo_trainer,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer
from utils import contiguous_copy

import wandb
from datasets import Dataset, load_from_disk


@dataclass
class EvaluateConfig(TrainingConfig):
    batch_size: int = 1
    learning_rate: float = 1.41e-6
    model_name: str = "gpt2"
    log_with: str = "wandb"
    base_models: List[str] = field(default_factory=lambda: ["gpt2"])
    ratio_threshold: float = 5.0
    use_score_scaling: bool = True
    use_score_norm: bool = True
    whiten_rewards: bool = True
    kl_penalty: str = "abs"
    mini_batch_size: int = 1
    init_kl_coef: float = 0.0
    entropy_coef: float = 1e-3
    prefix_length: int = 15
    continuation_length: int = 20
    continuation_max_str_length: int = 400
    dataset: str = "sentiment_prompts/negative_prompts_pos"
    project_name: str = "bbc"
    tracker_kwargs: Dict = field(default_factory=dict)
    prefix_gen_kwargs: Dict = field(
        default_factory=lambda: {
            "min_length": -1,
            "top_p": 1.0,
            "do_sample": True,
            "output_scores": True,
        }
    )
    continuation_gen_kwargs: Dict = field(
        default_factory=lambda: {
            "min_length": -1,
            "top_p": 0.9,
            "do_sample": True,
            "output_scores": True,
        }
    )
    run_id: str = ""
    num_generations: int = 1


@record
def evaluate(
    ppo_trainer: PPOTrainer,
    base_models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    reward_models: List[RewardModel],
    test_datasets: List[Dataset],
    logger: Logger,
    config: EvaluateConfig,
) -> None:
    """
    Train the policy model to control a set of base models using the given reward models
        and training dataset.

    Args:
        ppo_trainer (PPOTrainer): Used for logging and prefix generation.
        base_models (List[AutoModelForCausalLM]): Models to be controlled.
        tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
            base model.
        reward_model (List[RewardModel]): The reward model used for training.
        test_datasets (List[Dataset]): An evaluation dataset.
        logger (Logger): The logger instance for logging.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        None
    """
    try:
        # Pre-training setup
        base_models = ppo_trainer.accelerator.prepare(base_models)
        reward_models = [
            model.to(ppo_trainer.accelerator.device) for model in reward_models
        ]

        test_table = wandb.Table(
            columns=[
                "Dataset",
                "Reward",
                "Accuracy",
                "Perplexity",
                "Unigram",
                "Bigram",
                "Trigram",
            ]
        )
        log_dir = os.path.join(os.getcwd(), "local_logs", config.run_id)
        os.makedirs(log_dir, exist_ok=True)

        for test_dataset in test_datasets:
            # Create a directory for logs if it doesn't exist
            # Create a unique log file name
            process_index = ppo_trainer.accelerator.process_index
            log_file = os.path.join(
                log_dir, f"{test_dataset.info.dataset_name}_log_{process_index}.csv"
            )

            # Open the CSV file for writing
            with open(log_file, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile, escapechar="\\")

                # Write the header
                csv_writer.writerow(
                    [
                        "Epoch",
                        "Batch",
                        "Prefix",
                        "Prompt",
                        "Reward Type",
                        "Model Type",
                        "Continuation",
                        "Target Label",
                        "Reward",
                        "Correct",
                        "Perplexity",
                    ]
                )
                dataloader = DataLoader(
                    test_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    collate_fn=collator,
                )
                dataloader = ppo_trainer.accelerator.prepare(dataloader)

                # Test loop
                for batch_num, batch in enumerate(dataloader):
                    prefixes = generate_prefix(batch, ppo_trainer, config)
                    prefixes = contiguous_copy(prefixes, config.num_generations)
                    prompts = ppo_trainer.tokenizer.batch_decode(batch["prompt"])
                    prompts = contiguous_copy(prompts, config.num_generations)
                    prefix_prompt = [
                        prefix + prompt for prefix, prompt in zip(prefixes, prompts)
                    ]
                    rewards, continuations = compute_reward(
                        prompts,
                        prefix_prompt,
                        base_models,
                        tokenizers,
                        reward_models,
                        config,
                    )
                    batch["target"] = contiguous_copy(
                        batch["target"], config.num_generations
                    )
                    batch["target_label"] = contiguous_copy(
                        batch["target_label"], config.num_generations
                    )
                    targets = torch.tensor(batch["target"])
                    accuracy = (rewards.argmax(-1) == targets).long()
                    target_rewards = torch.gather(
                        rewards.mean(0).mean(0), -1, targets.unsqueeze(1)
                    )
                    target_rewards = [r for r in target_rewards]

                    base_model_perplexity = perplexity(
                        prompts, continuations, base_models, tokenizers
                    )

                    # Write detailed logs to CSV
                    local_log(
                        reward_models,
                        rewards,
                        accuracy,
                        continuations,
                        base_models,
                        prefixes,
                        prompts,
                        batch,
                        csv_writer,
                        0,
                        batch_num,
                        csvfile,
                        base_model_perplexity,
                    )
                    if batch_num % 10 == 0 and ppo_trainer.accelerator.is_main_process:
                        available = (
                            psutil.virtual_memory().available
                            * 100
                            / psutil.virtual_memory().total
                        )
                        print(f" Batch: {batch_num} \t RAM available: {available:.3f}%")

            ppo_trainer.accelerator.wait_for_everyone()
            if ppo_trainer.accelerator.is_main_process:
                import glob

                import pandas as pd

                run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id

                all_files = glob.glob(
                    os.path.join(log_dir, f"{test_dataset.info.dataset_name}_log_*")
                )
                combined_files = pd.concat([pd.read_csv(f) for f in all_files])

                for f in all_files:
                    os.remove(f)
                log_file_combined = os.path.join(
                    log_dir, f"{test_dataset.info.dataset_name}_log_{run_id}.csv"
                )
                combined_files.to_csv(log_file_combined, index=False)

                print(f"Detailed logs saved to {log_file_combined}")

                unigram = []
                bigram = []
                trigram = []

                num_chunks = len(combined_files) // config.num_generations
                for chunk in np.array_split(combined_files["Continuation"], num_chunks):
                    u, b, t = distinctness(chunk)
                    unigram.append(u)
                    bigram.append(b)
                    trigram.append(t)

                reward_mean = np.mean(combined_files["Reward"])
                reward_error = margin_of_error(combined_files["Reward"])
                accuracy_mean = np.mean(combined_files["Correct"])
                lo, hi = binomial_ci(combined_files["Correct"])
                perplexity_mean = np.mean(combined_files["Perplexity"])
                perplexity_error = margin_of_error(combined_files["Perplexity"])
                unigram_mean = np.mean(unigram)
                unigram_error = margin_of_error(unigram)
                bigram_mean = np.mean(bigram)
                bigram_error = margin_of_error(bigram)
                trigram_mean = np.mean(trigram)
                trigram_error = margin_of_error(trigram)
                test_table.add_data(
                    test_dataset.info.dataset_name,
                    f"{reward_mean:.3f} +- {reward_error:.3f}",
                    f"{accuracy_mean * 100:.3f}% ({lo * 100:.3f}, {hi * 100:.3f})",
                    f"{perplexity_mean:.3f} +- {perplexity_error:.3f}",
                    f"{unigram_mean:.3f} +- {unigram_error:.3f}",
                    f"{bigram_mean:.3f} +- {bigram_error:.3f}",
                    f"{trigram_mean:.3f} +- {trigram_error:.3f}",
                )
        ppo_trainer.accelerator.log({"Evaluation results": test_table})

        return None

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())

        return None


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def binomial_ci(samples: List[bool]):
    k = sum(samples)
    n = len(samples)
    result = binomtest(k, n)
    low, high = result.proportion_ci()
    return low, high


def margin_of_error(samples, confidence=0.95):
    se = stats.sem(samples)
    z_value = stats.norm.ppf((1 + confidence) / 2)
    margin = z_value * se
    return margin


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--run_id", type=str, help="WandB Run ID from training")
    parser.add_argument(
        "--base_models", nargs="+", default=["gpt2"], help="Base models"
    )
    parser.add_argument("--policy_model", type=str, default="gpt2", help="Policy model")
    parser.add_argument("--description", type=str, default="", help="Run description")
    parser.add_argument(
        "--dataset", type=str, default="imdb_sst2_processed", help="Dataset name"
    )
    args = parser.parse_args()
    # Set seed
    seed = 0
    torch.manual_seed(seed)
    # Initialize variables
    config = EvaluateConfig(
        policy_model=args.policy_model, base_models=args.base_models
    )
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.policy_model
    )
    config.policy_model = "gpt2"
    base_models = []
    base_model_tokenizers = []
    for base_model_name in config.base_models:
        base_models.append(AutoModelForCausalLM.from_pretrained(base_model_name))
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        base_model_tokenizers.append(tokenizer)

    test_file_names = [
        # "positive_prompts_neg",
        # "neutral_prompts_neg",
        "neutral_prompts_pos",
        # "negative_prompts_pos",
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
    reward_model = SentimentRewardModel()
    logger = Logger(__name__)
    ppo_trainer = prepare_ppo_trainer(policy_model, test_datasets[0], config)
    if ppo_trainer.accelerator.is_main_process:
        run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
        start = time.time()
    else:
        run_id = None

    run_id = broadcast_object_list([run_id])[0]
    print(f"Process ID: {ppo_trainer.accelerator.process_index}: Run ID : {run_id}")
    config.run_id = run_id

    evaluate(
        ppo_trainer,
        base_models,
        base_model_tokenizers,
        [reward_model],
        test_datasets,
        logger,
        config,
    )
