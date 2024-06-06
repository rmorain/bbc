from dataclasses import dataclass
from logging import Logger
from typing import Dict, Iterable, List, Optional

import pudb
import wandb
from reward_models import RewardModel, SentimentRewardModel
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from utils import BBCDataset, collator


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
    reward_model: Iterable[RewardModel],
    train_dataset: BBCDataset,
    logger: Logger,
    wandb_run: wandb.run,
    config: TrainingConfig,
) -> Optional[AutoModelForCausalLMWithValueHead]:
    """
    Train the policy model using the given reward model and training dataset.

    Args:
        policy_model (AutoModelForCausalLMWithValueHead): The model to be trained.
        reward_model (Iterable[RewardPipeline]): The reward model used for training.
        train_dataset (Dataset): The training dataset.
        logger (Logger): The logger instance for logging.
        wandb_run (wandb.run): The Weights and Biases run object for logging.
        config (TrainingConfig): The configuration object containing hyperparameters.

    Returns:
        Optional[AutoModelForCausalLMWithValueHead]: The trained policy model, or None if training failed.
    """
    try:
        # Pre-training setup
        ppo_trainer = prepare_ppo_trainer(policy_model, train_dataset, config)

        # Training loop
        for epoch in range(config.num_epochs):
            for batch in ppo_trainer.dataloader:
                prefix = generate_prefix(batch, ppo_trainer)
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
    train_dataset: BBCDataset,
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
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ppo_trainer = PPOTrainer(
        ppo_config, policy_model, tokenizer, train_dataset, data_collator=collator
    )
    return ppo_trainer


def generate_prefix(
    batch: Dict, ppo_trainer: PPOTrainer, gen_kwargs: Optional[Dict] = {}
):
    """
    Generate a prefix for each element of the batch.

    Args:
        batch (Dict): Batch of data.
        ppo_trainer (PPOTrainer): `PPOTrainer` object from the `trl` library.
        gen_kwargs (Optional[Dict]): Generation keyword arguments
    """
    pu.db
    pass


if __name__ == "__main__":
    # Initialize variables
    config = TrainingConfig()
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = BBCDataset("/home/rmorain2/bbc/datasets/imdb_sst2", tokenizer)
    reward_model = SentimentRewardModel()
    logger = Logger(__name__)
    run = wandb.init(project="bbc", config=dict(config))
    policy_model = train(
        policy_model, [reward_model], train_dataset, logger, run, config
    )
