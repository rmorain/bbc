import argparse
import csv
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from typing import List, Optional

import pudb
import torch
from reward_models import RewardModel, SentimentRewardModel
from train import compute_reward, generate_prefix, prepare_ppo_trainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset, load_from_disk
from trl import AutoModelForCausalLMWithValueHead


@dataclass
class EvaluateConfig:
    batch_size: int = 256
    learning_rate: float = 1.41e-6
    model_name: str = "gpt2"
    log_with: str = "wandb"
    ratio_threshold: float = 5.0
    use_score_scaling: bool = True
    use_score_norm: bool = True
    whiten_rewards: bool = True
    kl_penalty: str = "abs"
    mini_batch_size: int = 32
    init_kl_coef: float = 0.0
    entropy_coef: float = 1e-3
    prefix_length: int = 15
    continuation_length: int = 20
    continuation_max_str_length: int = 400
    dataset: str = "test_total"
    project_name: str = "bbc"


def evaluate(
    policy_model: AutoModelForCausalLMWithValueHead,
    base_models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    reward_models: List[RewardModel],
    test_dataset: Dataset,
    logger: Logger,
    config: EvaluateConfig,
) -> Optional[AutoModelForCausalLMWithValueHead]:
    """
    Train the policy model to control a set of base models using the given reward models
        and training dataset.

    Args:
        policy_model (AutoModelForCausalLMWithValueHead): The model to be trained.
        base_models (List[AutoModelForCausalLM]): Models to be controlled.
        tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
            base model.
        reward_model (List[RewardModel]): The reward model used for training.
        test_dataset (Dataset): The evaluation dataset.
        logger (Logger): The logger instance for logging.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        Optional[AutoModelForCausalLMWithValueHead]: The trained policy model, or None
            if training failed.
    """
    try:
        # Create a directory for logs if it doesn't exist
        pu.db
        log_dir = os.path.join(os.getcwd(), "local_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"eval_log_{timestamp}.csv")

        # Open the CSV file for writing
        with open(log_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write the header
            csv_writer.writerow(
                [
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
            ppo_trainer = prepare_ppo_trainer(policy_model, test_dataset, config)
            base_models = ppo_trainer.accelerator.prepare(base_models)
            reward_models = [
                model.to(ppo_trainer.accelerator.device) for model in reward_models
            ]

            # Test loop
            for batch_num, batch in enumerate(ppo_trainer.dataloader):
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

                prefix_prompt_ids = ppo_trainer.tokenizer(prefix_prompt).input_ids
                prefix_prompt_ids = [torch.tensor(ids) for ids in prefix_prompt_ids]
                stats = {}
                stats["env/accuracy"] = (
                    torch.mean(accuracy.float()).cpu().numpy().item()
                )
                stats["env/perplexity"] = perplexity
                stats["env/distinctness-unigram"] = diversity[0]
                stats["env/distinctness-bigram"] = diversity[1]
                stats["env/distinctness-trigram"] = diversity[2]

                batch["prefix"] = prefixes
                batch["prompt"] = prompts
                ppo_trainer.log_stats(
                    stats,
                    batch,
                    target_rewards,
                    columns_to_log=[],
                )
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

            logger.info(f"Detailed logs saved to {log_file}")

            return policy_model

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())

        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
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
    test_dataset = load_from_disk(os.environ.get("DATASETS_PATH") + config.dataset)
    if args.debug:
        debug_batch_size = 8
        test_dataset = test_dataset.select(range(debug_batch_size * 2))
        config.batch_size = debug_batch_size
        config.mini_batch_size = debug_batch_size
        config.project_name = "bbc-test"
    reward_model = SentimentRewardModel()
    logger = Logger(__name__)

    policy_model = evaluate(
        policy_model,
        [base_model],
        base_model_tokenizers,
        [reward_model],
        test_dataset,
        logger,
        config,
    )
