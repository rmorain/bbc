from dataclasses import dataclass
from logging import Logger
from typing import Iterable, Optional

import pudb
import wandb
from datasets import Dataset, load_from_disk
from pipelines import RewardPipeline
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from utils import collator


class BBCDataset(Dataset):
    def __init__(
        self,
        ds_path,
        emotions=["negative", "positive"],
        tokenizer=None,
    ):
        self.ds_path = ds_path
        self.emotions = emotions
        self.tokenizer = tokenizer
        try:
            self.ds = load_from_disk(self.ds_path + "_tokenized")
        except FileNotFoundError:
            self.ds = self._build_dataset()
            self.ds.save_to_disk(self.ds_path + "_tokenized")

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        ds = load_from_disk(self.ds_path)

        self.input_size = LengthSampler(input_min_text_length, input_max_text_length)
        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(["text", "label"])

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        if sample["label"] == 0:
            sample["target"] = 1
        else:
            sample["target"] = 0
        sample["target_label"] = self.emotions[sample["target"]]
        input_size = self.input_size()
        sample["prompt"] = self.tokenizer.encode(sample["text"])[:input_size]
        sample["query"] = self.tokenizer.encode(
            f"Sentiment: {self.emotions[sample['target']]}. {self.tokenizer.decode(sample['prompt'])}"
        )
        return sample

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def save_to_disk(self, path):
        self.ds.save_to_disk(path)


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
    reward_model: Iterable[RewardPipeline],
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
                pu.db
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
    Initialize the `PPOTrainer` class from the `trl` library using the `PPOConfig` class.

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
    ppo_trainer = PPOTrainer(
        ppo_config, policy_model, tokenizer, train_dataset, data_collator=collator
    )
    return ppo_trainer


def generate_prefix(batch, ppo_trainer):
    pass


if __name__ == "__main__":
    config = TrainingConfig()
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = BBCDataset("/home/rmorain2/bbc/datasets/imdb_sst2", tokenizer)
