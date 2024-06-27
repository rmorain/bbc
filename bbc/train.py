import argparse
import os
import traceback
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from reward_models import RewardModel, SentimentRewardModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from utils import collator

import wandb
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
    continuation_max_str_length: int = 400


def train(
    policy_model: AutoModelForCausalLMWithValueHead,
    base_models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    reward_models: List[RewardModel],
    train_dataset: Dataset,
    logger: Logger,
    config: TrainingConfig,
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
        train_dataset (Dataset): The training dataset.
        logger (Logger): The logger instance for logging.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        Optional[AutoModelForCausalLMWithValueHead]: The trained policy model, or None
            if training failed.
    """
    try:
        # Pre-training setup
        ppo_trainer = prepare_ppo_trainer(policy_model, train_dataset, config)
        base_models = ppo_trainer.accelerator.prepare(base_models)
        logger.info(base_models)
        reward_models = [
            model.to(ppo_trainer.accelerator.device) for model in reward_models
        ]

        # Training loop
        for _ in range(config.num_epochs):
            for batch in ppo_trainer.dataloader:
                prefixes = generate_prefix(batch, ppo_trainer, config)
                prompts = ppo_trainer.tokenizer.batch_decode(batch["prompt"])
                prefix_prompt = [
                    prefix + prompt for prefix, prompt in zip(prefixes, prompts)
                ]
                rewards, perplexity, diversity = compute_reward(
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

                batch["response"] = prefixes
                batch["query"] = ppo_trainer.tokenizer.batch_decode(batch["query"])
                ppo_trainer.log_stats(stats, batch, target_rewards)

        return policy_model

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
        prompts, continuation, base_models, reward_models, tokenizers
    )
    rewards = F.softmax(scores, dim=-1)
    return rewards, mean_perplexity, diversity


def generate_continuation(
    prefix_prompt: List[str],
    base_models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    config: TrainingConfig,
    gen_kwargs: Optional[Dict] = {
        "min_length": -1,
        "top_p": 1.0,
        "do_sample": False,
        "output_scores": True,
    },
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
        gen_kwargs (Optional[Dict]): Generation keyword arguments

    Returns:
        List[List[str]]: A list (len(base_models)) of lists (batch size) of
            continuation strings.
    """
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
    tokenizers: List[AutoTokenizer],
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
    # TODO: use prompts instead
    prompt_continuations = []
    # For base models
    for base_model_continuations, tokenizer in zip(continuation, tokenizers):
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

    # Initialize variables
    config = TrainingConfig()
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    base_model_tokenizers = [
        AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    ]
    for t in base_model_tokenizers:
        t.pad_token = t.eos_token
    train_dataset = load_from_disk(
        os.environ.get("DATASETS_PATH") + "imdb_sst2_tokenized"
    )
    if args.debug:
        debug_batch_size = 8
        train_dataset = train_dataset.select(range(debug_batch_size * 2))
        config.batch_size = debug_batch_size
        config.mini_batch_size = debug_batch_size
    reward_model = SentimentRewardModel()
    logger = Logger(__name__)

    wandb.require("core")
    if args.debug:
        run = wandb.init(project="bbc-test", config=asdict(config))
    else:
        run = wandb.init(project="bbc", config=asdict(config))

    policy_model = train(
        policy_model,
        [base_model],
        base_model_tokenizers,
        [reward_model],
        train_dataset,
        logger,
        config,
    )
