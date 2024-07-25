import argparse
import csv
import os
import traceback
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List, Optional

import torch
from reward_models import RewardModel, SentimentRewardModel
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from utils import collator

from datasets import Dataset, load_from_disk


@dataclass
class TrainingConfig:
    num_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1.41e-6
    policy_model: str = "gpt2"
    base_models: List[str] = field(default_factory=lambda: ["gpt2"])
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
    dataset: str = "imdb_sst2_processed"
    project_name: str = "bbc"
    tracker_kwargs: Dict = field(default_factory=dict)
    prefix_gen_kwargs: Dict = field(
        default_factory=lambda: {
            "min_length": -1,
            "top_p": 0.9,
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


@record
def train(
    ppo_trainer: PPOTrainer,
    base_models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    reward_models: List[RewardModel],
    logger: Logger,
    config: TrainingConfig,
) -> Optional[PPOTrainer]:
    """
    Train the policy model to control a set of base models using the given reward models
        and training dataset.

    Args:
        ppo_trainer (PPOTrainer): Used for logging, prefix generation, and updating the
            policy model.
        policy_model (AutoModelForCausalLMWithValueHead): The model to be trained.
        base_models (List[AutoModelForCausalLM]): Models to be controlled.
        tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
            base model.
        reward_model (List[RewardModel]): The reward model used for training.
        logger (Logger): The logger instance for logging.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        Optional[PPOTrainer]: The PPOTrainer object that contains the policy model, or
            None if training failed.
    """

    try:
        # Create a directory for logs if it doesn't exist
        log_dir = os.path.join(os.getcwd(), "local_logs", config.run_id)
        os.makedirs(log_dir, exist_ok=True)
        # Create a unique log file name
        process_index = ppo_trainer.accelerator.process_index
        log_file = os.path.join(log_dir, f"training_log_{process_index}.csv")

        # Open the CSV file for writing
        with open(log_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write the header
            csv_writer.writerow(
                [
                    "Epoch",
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

            # Training loop
            for epoch in range(config.num_epochs):
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

                    prefix_ids = ppo_trainer.tokenizer(prefixes).input_ids
                    prefix_prompt_ids = ppo_trainer.tokenizer(prefix_prompt).input_ids
                    prefix_prompt_ids = [torch.tensor(ids) for ids in prefix_prompt_ids]
                    mask = prefix_prompt_mask(prefix_ids, prefix_prompt_ids)
                    stats = ppo_trainer.step(
                        batch["query"], prefix_prompt_ids, target_rewards, mask
                    )
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
                                    epoch,
                                    batch_num,
                                    prefix,
                                    prompt,
                                    base_model.config._name_or_path,
                                    continuation,
                                    target,
                                    reward.item(),
                                    correct.item(),
                                ]
                            )

                    # Flush the CSV file to ensure data is written
                    csvfile.flush()

        ppo_trainer.accelerator.wait_for_everyone()
        if ppo_trainer.accelerator.is_main_process:
            import glob

            import pandas as pd

            run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
            path = os.path.join(log_dir, "training_log_*")

            all_files = glob.glob(path)
            combined_files = pd.concat([pd.read_csv(f) for f in all_files])

            for f in all_files:
                os.remove(f)
            log_file_combined = os.path.join(log_dir, f"training_log_{run_id}.csv")
            combined_files.to_csv(log_file_combined, index=False)

            print(f"Detailed logs saved to {log_file_combined}")

        ppo_trainer.accelerator.wait_for_everyone()
        return ppo_trainer

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())

        return None


def prepare_ppo_trainer(
    policy_model: AutoModelForCausalLMWithValueHead,
    train_dataset: Dataset,
    config: TrainingConfig,
) -> PPOTrainer:
    """
    Initialize the `PPOTrainer` class from the `trl` library using the `PPOConfig`
        class.

    Args:
        policy_model (AutoModelForCausalLMWithValueHead): The model to be trained.
        train_dataset (Dataset): The training dataset.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        PPOTrainer: The object that updates the policy model
    """
    ppo_config = PPOConfig(
        model_name=config.policy_model,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        log_with=config.log_with,
        ratio_threshold=config.ratio_threshold,
        use_score_scaling=config.use_score_scaling,
        use_score_norm=config.use_score_norm,
        whiten_rewards=config.whiten_rewards,
        kl_penalty=config.kl_penalty,
        mini_batch_size=config.mini_batch_size,
        init_kl_coef=config.init_kl_coef,
        entropy_coef=config.entropy_coef,
        remove_unused_columns=False,
        tracker_project_name=config.project_name,
        tracker_kwargs=config.tracker_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model)
    tokenizer.pad_token = tokenizer.eos_token
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )
    return ppo_trainer


def generate_prefix(
    batch: Dict,
    ppo_trainer: PPOTrainer,
    config: TrainingConfig,
) -> List[torch.LongTensor]:
    """
    Generate a prefix for each element of the batch.

    Args:
        batch (Dict): Batch of data.
        ppo_trainer (PPOTrainer): `PPOTrainer` object from the `trl` library.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        List[torch.LongTensor]: A list (batch size) of tensors containing prefix tokens.
    """
    gen_kwargs = config.prefix_gen_kwargs
    query_prefix = ppo_trainer.generate(
        batch["query"],
        max_new_tokens=config.prefix_length,
        pad_token_id=ppo_trainer.tokenizer.eos_token_id,
        **gen_kwargs,
    )
    prefix = [
        query_prefix[i][len(batch["query"][i]) :] for i in range(len(query_prefix))
    ]
    prefix_str = ppo_trainer.tokenizer.batch_decode(prefix)

    return prefix_str


def compute_reward(
    prompts: List[str],
    prefix_prompt: List[torch.LongTensor],
    base_models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    reward_models: List[RewardModel],
    config: TrainingConfig,
) -> List[float]:
    """
    Compute a reward for each (prompt, continuation) pair.

    Args:
        prompts (List[str]): A list (batch size) of prompt strings.
        prefix_prompt (List[torch.LongTensor]): A list (batch size) of prefix tensors
            prepended to prompt tensors.
        base_models (AutoModelForCausalLM): A list of language models to be controlled
            by the policy model.
        tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
            base model.
        reward_models (List[RewardModel]): A list of reward models.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        List[float]: A list (batch size) of reward values.
    """
    continuation = generate_continuation(prefix_prompt, base_models, tokenizers, config)
    mean_perplexity = perplexity(prompts, continuation, base_models, tokenizers)
    diversity = distinctness(continuation)
    scores = compute_scores(
        prompts,
        continuation,
        base_models,
        reward_models,
    )
    return scores, mean_perplexity, diversity, continuation


def generate_continuation(
    prefix_prompt: List[str],
    base_models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    config: TrainingConfig,
) -> List[List[str]]:
    """
    Generates a continuation from a (prefix, prompt) pair for each base model.

    Args:
        prefix_prompt (List[str]): A list (batch size) of prefix strings
            prepended to prompt strings.
        base_models (AutoModelForCausalLM): A list of language models to be controlled
            by the policy model.
        tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
            base model.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        List[List[str]]: A list (len(base_models)) of lists (batch size) of
            continuation strings.
    """
    gen_kwargs = config.continuation_gen_kwargs
    continuations = []
    with torch.no_grad():
        for model, tokenizer in zip(base_models, tokenizers):
            inputs = tokenizer(prefix_prompt, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            prefix_prompt_continuation = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.continuation_length,
                pad_token_id=model.config.eos_token_id,
                **gen_kwargs,
            )
            prefix_prompt_continuation_str = tokenizer.batch_decode(
                prefix_prompt_continuation,
                skip_special_tokens=True,
            )
            continuation = [
                s[len(pp) : config.continuation_max_str_length]
                for s, pp in zip(prefix_prompt_continuation_str, prefix_prompt)
            ]
            continuations.append(continuation)
    return continuations


def compute_scores(
    prompts: List[str],
    continuation: List[List[torch.LongTensor]],
    base_models: List[AutoModelForCausalLM],
    reward_models: List[RewardModel],
) -> List[List[float]]:
    """
    Computes a score from each (prompt, continuation) pair for each reward model.

    Args:
        prompts (List[str]): A list (batch size) of prompt strings.
        continuation List[List[torch.LongTensor]]: A list (len(base_models)) of lists
            (batch size) of tensors containing continuation tokens.
        base_models (AutoModelForCausalLM): A list of language models to be controlled
            by the policy model.
        reward_models (List[RewardModel]): A list of reward models.
        tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
            base model.

    Returns:
        List[float]: A list (batch size) of scores.
    """
    # (num base models * batch size)
    prompt_continuations = []
    # For base models
    for base_model_continuations in continuation:
        for prompt, continuation in zip(prompts, base_model_continuations):
            prompt_continuation = prompt + continuation
            prompt_continuations.append(prompt_continuation)
    # (num reward models, num base models * batch size, num classes)
    scores = []
    for model in reward_models:
        s = model(prompt_continuations)
        scores.append(s)
    scores_tensor = torch.tensor(scores).reshape(
        (len(reward_models), len(base_models), len(prompts), len(scores[0][0]))
    )

    # (batch size, num classes)
    mean_scores = scores_tensor.mean(0).mean(0)
    return mean_scores


def prefix_prompt_mask(
    prefix: List[torch.LongTensor], prefix_prompt: List[torch.LongTensor]
) -> List[torch.LongTensor]:
    """
    Generates a mask that keeps prefix tokens and removes prompt tokens.

    Args:
        prefix (List[torch.LongTensor]): A list (batch size) of prefix tensors.
        prefix_prompt (List[torch.LongTensor]): A list (batch size) of prefix tensors
            prepended to prompt tensors.

    Returns:
        List[torch.LongTensor]: A list (batch size) of mask values for each (prefix,
            prompt) pair.
    """
    mask = []
    for p, r in zip(prefix, prefix_prompt):
        prefix_mask = torch.ones(len(p), dtype=torch.long)
        prompt_mask = torch.zeros(len(r) - len(p), dtype=torch.long)
        prefix_prompt_mask = torch.cat((prefix_mask, prompt_mask), dim=0)
        mask.append(prefix_prompt_mask)
    return mask


def perplexity(
    prompts: List[str],
    continuations: List[List[torch.LongTensor]],
    base_models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
) -> float:
    """
    Computes the perplexity of each continuation averaged across the number of base
        models.

    Args:
        prompts (List[str]): A list (batch size) of prompt strings.
        continuations (List[List[str]]): A list (len(base_models)) of lists
            (batch size) of tensors containing continuation strings.
        base_models (AutoModelForCausalLM): A list of language models to be controlled
            by the policy model.
        tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
            base model.

    Returns:
        float: Mean perplexity across base models and continuations.
    """
    losses = []
    for base_model, tokenizer, base_model_continuations in zip(
        base_models, tokenizers, continuations
    ):
        prompt_continuations = [
            p + c for p, c in zip(prompts, base_model_continuations)
        ]
        inputs = tokenizer(prompt_continuations, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(base_model.device)
        attention_mask = inputs.attention_mask.to(base_model.device)
        target_ids = input_ids.clone()
        continuation_ids = tokenizer(base_model_continuations).input_ids
        continuation_lengths = torch.tensor(
            [len(continuation) for continuation in continuation_ids]
        )
        for target, i in zip(target_ids, continuation_lengths):
            j = len(target) - i
            target[:j] = -100

        # make prompt ids in target ids -100
        outputs = base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=target_ids
        )
        losses.append(outputs.loss)
    perplexity = torch.tensor(losses).mean().exp().item()
    return perplexity


def distinctness(continuations: List[List[str]]) -> tuple[float]:
    """
    Evaluate the number of unique unigrams, bigrams, and trigrams in the list of
        strings.

    Args:
        continuations (List[List[str]]): A list (len(base_models)) of lists
            (batch size) of tensors containing continuation strings.
    """
    total_words = 0
    unigrams, bigrams, trigrams = set(), set(), set()

    for base_model_continuations in continuations:
        for continuation in base_model_continuations:
            o = continuation.split(" ")
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + "_" + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + "_" + o[i + 1] + "_" + o[i + 2])

    if total_words == 0:
        return 0.0, 0.0, 0.0

    dist1 = len(unigrams) / total_words
    dist2 = len(bigrams) / total_words
    dist3 = len(trigrams) / total_words

    return dist1, dist2, dist3


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    # Set seed
    seed = 0
    torch.manual_seed(seed)
    # Initialize variables
    config = TrainingConfig()
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.policy_model
    )
    base_model = AutoModelForCausalLM.from_pretrained(config.base_models[0])
    base_model_tokenizers = [
        AutoTokenizer.from_pretrained(config.base_models[0], padding_side="left")
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

    ppo_trainer = prepare_ppo_trainer(policy_model, train_dataset, config)

    ppo_trainer = train(
        ppo_trainer,
        policy_model,
        [base_model],
        base_model_tokenizers,
        [reward_model],
        logger,
        config,
    )
