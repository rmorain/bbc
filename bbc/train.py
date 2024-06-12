import argparse
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Dict, Iterable, List, Optional, Tuple

import pudb
import torch
import torch.nn.functional as F
import wandb
from reward_models import RewardModel, SentimentRewardModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from utils import collator

from datasets import Dataset, load_from_disk


@dataclass
class TrainingConfig:
    num_epochs: int = 1
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


def train(
    policy_model: AutoModelForCausalLMWithValueHead,
    base_models: Iterable[AutoModelForCausalLM],
    reward_models: Iterable[RewardModel],
    train_dataset: Dataset,
    logger: Logger,
    wandb_run: wandb.run,
    config: TrainingConfig,
) -> Optional[AutoModelForCausalLMWithValueHead]:
    """
    Train the policy model using the given reward model and training dataset.

    Args:
        policy_model (AutoModelForCausalLMWithValueHead): The model to be trained.
        base_models (Iterable[AutoModelForCausalLM]): Models to be controlled.
        reward_model (Iterable[RewardModel]): The reward model used for training.
        train_dataset (Dataset): The training dataset.
        logger (Logger): The logger instance for logging.
        wandb_run (wandb.run): The Weights and Biases run object for logging.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        Optional[AutoModelForCausalLMWithValueHead]: The trained policy model, or None
            if training failed.
    """
    try:
        # Pre-training setup
        ppo_trainer = prepare_ppo_trainer(policy_model, train_dataset, config)
        base_models = [ppo_trainer.accelerator.prepare(model) for model in base_models]
        reward_models = [
            model.to(ppo_trainer.accelerator.device) for model in base_models
        ]

        # Training loop
        for epoch in range(config.num_epochs):
            for batch in ppo_trainer.dataloader:
                prefix = generate_prefix(batch, ppo_trainer, config)
                prefix_prompt = [
                    torch.cat((prefix[i], batch["prompt"][i]))
                    for i in range(config.batch_size)
                ]
                rewards, accuracy = compute_reward(
                    batch,
                    prefix_prompt,
                    base_models,
                    reward_models,
                    ppo_trainer.tokenizer,
                    config,
                )
                mask = prefix_prompt_mask(prefix, prefix_prompt)
                stats = ppo_trainer.step(batch["query"], prefix_prompt, rewards, mask)
                logger.info(f"Epoch {epoch} - Loss: {loss_value}")
                wandb_run.log({"loss": loss_value})

        # Post-training tasks
        # ...

        return policy_model

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
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
        model_name=config.model_name,
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
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
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
    gen_kwargs: Optional[Dict] = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "output_scores": True,
    },
) -> List[torch.LongTensor]:
    """
    Generate a prefix for each element of the batch.

    Args:
        batch (Dict): Batch of data.
        ppo_trainer (PPOTrainer): `PPOTrainer` object from the `trl` library.
        config (TrainingConfig): The configuration object containing hyperparameters.
        gen_kwargs (Optional[Dict]): Generation keyword arguments

    Returns:
        List[torch.LongTensor]: A list (batch size) of tensors containing prefix tokens.
    """
    query_prefix = ppo_trainer.generate(
        batch["query"],
        max_new_tokens=config.prefix_length,
        pad_token_id=ppo_trainer.tokenizer.eos_token_id,
        **gen_kwargs,
    )
    prefix = [
        query_prefix[i][len(batch["query"][i]) :] for i in range(len(query_prefix))
    ]

    return prefix


def compute_reward(
    batch: Dict,
    prefix_prompt: List[torch.LongTensor],
    base_models: List[AutoModelForCausalLM],
    reward_models: List[RewardModel],
    tokenizer: AutoTokenizer,
    config: TrainingConfig,
) -> List[float]:
    """
    Compute a reward for each (prompt, continuation) pair.

    Args:
        batch (Dict): Batch of query, prompt, and target data.
        prefix_prompt (List[torch.LongTensor]): A list (batch size) of prefix tensors
            prepended to prompt tensors.
        base_models (AutoModelForCausalLM): A list of language models to be controlled
            by the policy model.
        reward_models (List[RewardModel]): A list of reward models.
        tokenizer (AutoTokenizer): A tokenizer.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        List[float]: A list (batch size) of reward values.
    """
    continuation = generate_continuation(prefix_prompt, base_models, config)
    scores = compute_scores(batch, continuation, reward_models, tokenizer)
    reward, accuracy = process_scores(scores, batch["target"], len(base_models))
    pass


def generate_continuation(
    prefix_prompt: List[torch.LongTensor],
    base_models: List[AutoModelForCausalLM],
    config: TrainingConfig,
    gen_kwargs: Optional[Dict] = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "output_scores": True,
    },
) -> List[List[torch.LongTensor]]:
    """
    Generates continuations from a (prefix, prompt) pair.

    Args:
        prefix_prompt (List[torch.LongTensor]): A list (batch size) of prefix tensors
            prepended to prompt tensors.
        base_models (AutoModelForCausalLM): A list of language models to be controlled
            by the policy model.
        config (TrainingConfig): The configuration object containing hyperparameters.
        gen_kwargs (Optional[Dict]): Generation keyword arguments

    Returns:
        List[List[torch.LongTensor]]: A list (len(base_models)) of lists (batch size) of
            tensors containing continuation tokens.
    """
    continuations = []
    with torch.no_grad():
        for i, model in enumerate(base_models):
            continuations.append([])
            for x in prefix_prompt:
                # attention_mask = torch.ones_like(x)
                prefix_prompt_continuation = model.generate(
                    x.unsqueeze(0),
                    max_new_tokens=config.continuation_length,
                    # attention_mask=attention_mask,
                    pad_token_id=model.config.eos_token_id,
                    **gen_kwargs,
                ).squeeze(0)
                continuation = prefix_prompt_continuation[len(x) :]
                continuations[i].append(continuation)
    return continuations


def compute_scores(
    batch: Dict,
    continuation: List[List[torch.LongTensor]],
    base_models: List[AutoModelForCausalLM],
    reward_models: List[RewardModel],
    tokenizer: AutoTokenizer,
) -> List[List[float]]:
    """
    Compute scores for each (prompt, continuation) pair.

    Args:
        batch (Dict): Batch of query, prompt, and target data.
        continuation List[List[torch.LongTensor]]: A list (len(base_models)) of lists
            (batch size) of tensors containing continuation tokens.
        base_models (AutoModelForCausalLM): A list of language models to be controlled
            by the policy model.
        reward_models (List[RewardModel]): A list of reward models.
        tokenizer (AutoTokenizer): A tokenizer.

    Returns:
        List[float]: A list (batch size) of scores.
    """
    # (num base models * batch size)
    prompt_continuation = []
    # For base models
    for base_model in continuation:
        for p, c in zip(batch["prompt"], base_model):
            pc = torch.cat((p, c))
            pc_str = tokenizer.decode(pc)
            prompt_continuation.append(pc_str)
    # (num reward models, num base models * batch size, num classes)
    scores = []
    for model in reward_models:
        s = model(prompt_continuation)
        scores.append(s)
    scores_tensor = torch.tensor(scores).reshape(
        (len(reward_models, len(base_models), len(batch["prompt"]), len(scores[0][0])))
    )
    # (batch size, num classes)
    mean_scores = scores_tensor.mean(0).mean(0)
    reward = F.softmax(mean_scores, dim=1)
    accuracy = mean_scores.argmax(1) == batch["target"]
    return reward, accuracy


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


def process_scores(
    class_scores: List[List[float]], target: List[int], num_base_models: int
) -> Tuple[List[float], List[bool]]:
    """
    Processes the raw scores into reward values.

    Args:
        class_scores (List[List[List[float]]]): A List (batch size * number of base models) of
            Lists (number of classes) containing scores.
        target (List[int]): A List (batch size) of indices of the correct class.
        num_base_models (int): The number of base models used to create class_scores.

    Returns:
        Tuple[List[float], List[bool]]: A tuple containing two lists:
            - List (batch size * number of base models) of reward values
            - List (batch size * number of base models) of accuracy successes and
                failures.
    """
    scores = torch.tensor(class_scores)
    # (batch size, num base models, num classes)
    scores = scores.reshape((len(target), num_base_models, len(class_scores[0])))
    rewards = F.softmax(scores, dim=2)
    predictions = torch.argmax(scores, dim=2)  # (batch size, num base models)
    target = torch.tensor(target).unsqueeze(1)
    accuracy = predictions == target
    return rewards, accuracy


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Initialize variables
    config = TrainingConfig()
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_from_disk("/home/rmorain2/bbc/datasets/imdb_sst2_tokenized")
    if args.debug:
        debug_batch_size = 8
        train_dataset = train_dataset.select(range(debug_batch_size))
        config.batch_size = debug_batch_size
        config.mini_batch_size = debug_batch_size
    reward_model = SentimentRewardModel()
    logger = Logger(__name__)
    run = wandb.init(project="bbc", config=asdict(config))
    policy_model = train(
        policy_model, [base_model], [reward_model], train_dataset, logger, run, config
    )
