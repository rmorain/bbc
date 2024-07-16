import argparse
import csv
import os
import traceback
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List

import numpy as np
import scipy.stats as stats
import torch
from reward_models import RewardModel, SentimentRewardModel
from scipy.stats import binomtest
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from train import compute_reward, generate_prefix, prepare_ppo_trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer

import wandb
from datasets import Dataset, load_from_disk


@dataclass
class EvaluateConfig:
    batch_size: int = 1
    learning_rate: float = 1.41e-6
    model_name: str = "gpt2"
    log_with: str = "wandb"
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
        # Create a directory for logs if it doesn't exist
        run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
        log_dir = os.path.join(os.getcwd(), "local_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_dir = os.path.join(log_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file name
        process_id = ppo_trainer.accelerator.process_index
        run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
        log_file = os.path.join(log_dir, f"eval_log_{run_id}_{process_id}.csv")

        # Open the CSV file for writing
        with open(log_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write the header
            csv_writer.writerow(
                [
                    "Dataset",
                    "Batch",
                    "Prefix",
                    "Prompt",
                    "Model Type",
                    "Continuation",
                    "Target Label",
                    "Reward",
                    "Correct",
                ]
            )

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

            for test_dataset in test_datasets:
                dataloader = DataLoader(
                    test_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    collate_fn=collator,
                )
                reward_list = []
                accuracy_list = []
                perplexity_list = []
                unigram_list = []
                bigram_list = []
                trigram_list = []

                # Test loop
                for batch_num, batch in enumerate(dataloader):
                    prefixes = generate_prefix(batch, ppo_trainer, config)
                    prompts = ppo_trainer.tokenizer.batch_decode(batch["prompt"])
                    prefix_prompt = [
                        prefix + prompt for prefix, prompt in zip(prefixes, prompts)
                    ]
                    rewards, perplexity, diversity, continuations = compute_reward(
                        prompts,
                        prefix_prompt,
                        base_models,
                        tokenizers,
                        reward_models,
                        config,
                    )
                    targets = torch.tensor(batch["target"])
                    accuracy = (rewards.argmax(-1) == targets).long()
                    target_rewards = torch.gather(rewards, 1, targets.unsqueeze(1))
                    target_rewards = [r for r in target_rewards]
                    reward_list.append(target_rewards[0].item())
                    accuracy_list.append(accuracy[0].item())
                    perplexity_list.append(perplexity)
                    unigram_list.append(diversity[0])
                    bigram_list.append(diversity[1])
                    trigram_list.append(diversity[2])

                    # Write detailed logs to CSV
                    for base_model_continuation, base_model in zip(
                        continuations, base_models
                    ):
                        for (
                            prefix,
                            prompt,
                            continuation,
                            target,
                            reward,
                            correct,
                        ) in zip(
                            prefixes,
                            prompts,
                            base_model_continuation,
                            batch["target_label"],
                            target_rewards,
                            accuracy,
                        ):
                            csv_writer.writerow(
                                [
                                    test_dataset.info.dataset_name,
                                    batch_num,
                                    prefix,
                                    prompt,
                                    base_model.config.model_type,
                                    continuation,
                                    target,
                                    reward.item(),
                                    correct.item(),
                                ]
                            )

                    # Flush the CSV file to ensure data is written
                    csvfile.flush()

                reward_mean = np.mean(reward_list)
                reward_error = margin_of_error(reward_list)
                accuracy_mean = np.mean(accuracy_list)
                lo, hi = binomial_ci(accuracy_list)
                perplexity_mean = np.mean(perplexity_list)
                perplexity_error = margin_of_error(perplexity_list)
                unigram_mean = np.mean(unigram_list)
                unigram_error = margin_of_error(unigram_list)
                bigram_mean = np.mean(bigram_list)
                bigram_error = margin_of_error(bigram_list)
                trigram_mean = np.mean(trigram_list)
                trigram_error = margin_of_error(trigram_list)
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
            logger.info(f"Detailed logs saved to {log_file}", main_process_only=True)

            if ppo_trainer.accelerator.is_main_process:
                import glob

                import pandas as pd

                all_files = glob.glob(os.path.join(log_dir, f"eval_log_{run_id}*"))
                combined_files = pd.concat([pd.read_csv(f) for f in all_files])

                for f in all_files:
                    os.remove(f)
                log_file_combined = os.path.join(log_dir, f"eval_log_{run_id}.csv")
                combined_files.to_csv(log_file_combined, index=False)

                print(f"Detailed logs saved to {log_file_combined}")

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
    args = parser.parse_args()
    # Set seed
    seed = 0
    torch.manual_seed(seed)
    # Initialize variables
    config = EvaluateConfig()
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    base_model_tokenizers = [
        AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    ]
    for t in base_model_tokenizers:
        t.pad_token = t.eos_token
    test_file_names = os.listdir(os.environ.get("DATASETS_PATH") + "sentiment_prompts")

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

    evaluate(
        ppo_trainer,
        policy_model,
        [base_model],
        base_model_tokenizers,
        [reward_model],
        test_datasets,
        logger,
        config,
    )
