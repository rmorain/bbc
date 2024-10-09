import argparse
import csv
import os
import time
import traceback
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List, Optional

import psutil
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
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "output_scores": True,
        }
    )
    continuation_gen_kwargs: Dict = field(
        default_factory=lambda: {
            "min_length": -1,
            # "top_p": 0.9,
            "top_k": 0.0,
            "do_sample": False,
            "output_scores": True,
        }
    )
    run_id: str = ""
    signal_reset = False


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
                    "Reward Type",
                    "Model Type",
                    "Continuation",
                    "Target Label",
                    "Reward",
                    "Correct",
                    "Perplexity",
                ]
            )

            # Pre-training setup
            base_models = [base_model.cuda() for base_model in base_models]
            print([base_model.device for base_model in base_models])
            reward_models = [
                model.to(ppo_trainer.accelerator.device) for model in reward_models
            ]
            best_accuracy = 0.0

            # Training loop
            for epoch in range(config.num_epochs):
                epoch_accuracy = 0.0
                num_batches = 0
                if ppo_trainer.accelerator.is_main_process:
                    print(f"Begin epoch: {epoch}")
                    start = time.time()
                for batch_num, batch in enumerate(ppo_trainer.dataloader):
                    torch.cuda.empty_cache()
                    prefixes = generate_prefix(batch, ppo_trainer, config)
                    prompts = ppo_trainer.tokenizer.batch_decode(batch["prompt"])
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
                    targets = torch.tensor(batch["target"])
                    accuracy = (rewards.argmax(-1) == targets).long()
                    target_rewards = torch.gather(
                        rewards.mean(0).mean(0), -1, targets.unsqueeze(1)
                    )
                    target_rewards = [r for r in target_rewards]

                    prefix_ids = ppo_trainer.tokenizer(prefixes).input_ids
                    prefix_prompt_ids = [
                        torch.cat((torch.tensor(pre).to(pro.device), pro))
                        for pre, pro in zip(prefix_ids, batch["prompt"])
                    ]
                    mask = prefix_prompt_mask(prefix_ids, prefix_prompt_ids)
                    stats = ppo_trainer.step(
                        batch["query"], prefix_prompt_ids, target_rewards, mask
                    )
                    stats["env/accuracy"] = (
                        torch.mean(accuracy.float()).cpu().numpy().item()
                    )
                    # stats["env/perplexity"] = perplexity.flatten().mean(0).item()
                    # stats["env/distinctness-unigram"] = diversity[0]
                    # stats["env/distinctness-bigram"] = diversity[1]
                    # stats["env/distinctness-trigram"] = diversity[2]

                    batch["prefix"] = prefixes
                    batch["prompt"] = prompts
                    ppo_trainer.log_stats(
                        stats,
                        batch,
                        target_rewards,
                        columns_to_log=[],
                    )
                    # Write detailed logs to CSV
                    dummy_perplexity = torch.ones((len(continuations), len(prompts)))
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
                        epoch,
                        batch_num,
                        csvfile,
                        dummy_perplexity,
                    )
                    epoch_accuracy += stats["env/accuracy"]
                    num_batches += 1
                    if batch_num % 10 == 0 and ppo_trainer.accelerator.is_main_process:
                        available = (
                            psutil.virtual_memory().available
                            * 100
                            / psutil.virtual_memory().total
                        )
                        print(f" Batch: {batch_num} \t RAM available: {available:.3f}%")
                        gpu_memory = torch.cuda.max_memory_allocated() / 1e9
                        print(f"Max GPU memory:\t{gpu_memory:.3f} GB")
                    if config.signal_reset:
                        break
                if config.signal_reset:
                    break
                if ppo_trainer.accelerator.is_main_process:
                    avg_epoch_accuracy = epoch_accuracy / num_batches
                    # Check if this is the best model so far
                    if avg_epoch_accuracy > best_accuracy:
                        best_accuracy = avg_epoch_accuracy
                        # Save the best model
                        save_model(ppo_trainer, config)
                    print(f"End epoch {epoch} Duration: {time.time() - start}")
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
        print(f"Training failed: {str(e)}")
        print(traceback.format_exc())
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.accelerator.get_tracker("wandb").finish()
        process_index = ppo_trainer.accelerator.process_index
        gpu_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Max GPU memory {process_index}: {gpu_memory:.3f} GB")

        return None


def save_model(ppo_trainer, train_config):
    # Save policy model
    if ppo_trainer.accelerator.is_main_process:
        # Create a directory for saved models if it doesn't exist
        run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            # save_dir = os.path.join(os.getcwd(), "checkpoints", run_id)
            print("Not saving debug run")
            return None
        else:
            save_dir = os.path.join(os.getcwd(), "checkpoints", job_id, "policy_models")

        os.makedirs(save_dir, exist_ok=True)
        restart_count = int(os.getenv("SLURM_RESTART_COUNT", 0))
        model_dir = os.path.join(
            save_dir, f"{train_config.policy_model}_{run_id}_{restart_count}"
        )
        ppo_trainer.save_pretrained(model_dir)
        print(f"Policy model saved at {model_dir}")
        with open(
            os.path.join(os.getcwd(), "checkpoints", job_id, "model_name.txt"), "w"
        ) as f:
            f.write(model_dir)


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
        optimize_device_cache=True,
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
    prefix_str = ppo_trainer.tokenizer.batch_decode(prefix, skip_special_tokens=False)

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
    # base_model_perplexity = perplexity(prompts, continuation, base_models, tokenizers)
    scores = compute_scores_continuation_only(
        continuation,
        reward_models,
    )
    return scores, continuation


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
            continuation_ids = [
                cont_ids[len(pp_ids) :]
                for cont_ids, pp_ids in zip(prefix_prompt_continuation, input_ids)
            ]
            continuation_str = tokenizer.batch_decode(
                continuation_ids,
                skip_special_tokens=False,
            )
            continuation = [
                s[: config.continuation_max_str_length] for s in continuation_str
            ]
            if not all(continuation):
                print("ERROR: Missing continuations")
                print(continuation)
                print(prefix_prompt)
            continuations.append(continuation)
    return continuations


def compute_scores(
    prompts: List[str],
    continuations: List[List[torch.LongTensor]],
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
    for base_model_continuations in continuations:
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
    # mean_scores = scores_tensor.mean(0).mean(0)
    # mean_scores = scores_tensor.mean(0)
    # (reward_model, base models, batch size, num classes)
    return scores_tensor


def compute_scores_continuation_only(
    continuations: List[List[torch.LongTensor]],
    reward_models: List[RewardModel],
) -> List[List[float]]:
    scores = []
    for base_model_continuations in continuations:
        for model in reward_models:
            s = model(base_model_continuations)
            scores.append(s)
    scores_tensor = torch.tensor(scores).reshape(
        len(reward_models), len(continuations), len(continuations[0]), len(scores[0][0])
    )
    return scores_tensor


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
) -> List[float]:
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
        List[float]: Perplexity for each base models and continuations.
    """
    losses = []
    for base_model, tokenizer, base_model_continuations in zip(
        base_models, tokenizers, continuations
    ):
        losses.append([])
        for prompt, continuation in zip(prompts, base_model_continuations):
            prompt_continuation = prompt + continuation
            inputs = tokenizer(prompt_continuation, return_tensors="pt")
            input_ids = inputs.input_ids.to(base_model.device)
            # If eos token, only include one
            eos_index = (input_ids == tokenizer.eos_token_id).nonzero()
            if eos_index.numel() > 0:
                eos_index = eos_index[0, 1]
            else:
                eos_index = input_ids.shape[-1]
            input_ids = input_ids[:, : eos_index + 1]
            attention_mask = inputs.attention_mask.to(base_model.device)
            attention_mask = attention_mask[:, : eos_index + 1]
            target_ids = input_ids.clone()
            prompt_ids = tokenizer(prompt).input_ids
            target_ids[:, : len(prompt_ids)] = -100

            # make prompt ids in target ids -100
            outputs = base_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=target_ids
            )
            losses[-1].append(outputs.loss)
    perplexity = torch.tensor(losses).exp()
    return perplexity


def distinctness(continuations: List[str]) -> tuple[float]:
    """
    Evaluate the number of unique unigrams, bigrams, and trigrams in the list of
        strings.

    Args:
        continuations (List[str]): A list (len(base_models)) of lists
            (batch size) of tensors containing continuation strings.
    """
    total_words = 0
    unigrams, bigrams, trigrams = set(), set(), set()

    for continuation in continuations:
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


def local_log(
    reward_models,
    rewards,
    accuracy,
    continuations,
    base_models,
    prefixes,
    prompts,
    batch,
    csv_writer,
    epoch,
    batch_num,
    csvfile,
    perplexity,
):
    for reward_model, reward_model_reward, reward_model_accuracy in zip(
        reward_models, rewards, accuracy
    ):
        for (
            base_model_continuation,
            base_model,
            base_model_reward,
            base_model_accuracy,
            base_model_perplexity,
        ) in zip(
            continuations,
            base_models,
            reward_model_reward,
            reward_model_accuracy,
            perplexity,
        ):
            for (
                prefix,
                prompt,
                continuation,
                target_label,
                target,
                batch_reward,
                correct,
                batch_perplexity,
            ) in zip(
                prefixes,
                prompts,
                base_model_continuation,
                batch["target_label"],
                batch["target"],
                base_model_reward,
                base_model_accuracy,
                base_model_perplexity,
            ):
                csv_writer.writerow(
                    [
                        epoch,
                        batch_num,
                        prefix,
                        prompt,
                        reward_model.__class__.__name__,
                        base_model.config._name_or_path,
                        continuation,
                        target_label,
                        batch_reward[target].item(),
                        correct.item(),
                        batch_perplexity.item(),
                    ]
                )

    # Flush the CSV file to ensure data is written
    csvfile.flush()


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
