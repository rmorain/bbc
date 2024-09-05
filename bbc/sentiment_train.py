import argparse
import os
import signal
import time
from datetime import timedelta

import torch
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
from reward_models import SentimentRewardModel
from train import TrainingConfig, prepare_ppo_trainer, train
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from datasets import load_from_disk


def _reset_handler(sigint, stack_frame):
    global train_config
    train_config.signal_reset = True
    return


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--base_models", nargs="+", default=["gpt2"], help="Base models")
parser.add_argument("--policy_model", type=str, default="gpt2", help="Policy model")
parser.add_argument("--description", type=str, default="", help="Run description")
parser.add_argument(
    "--dataset", type=str, default="imdb_sst2_processed", help="Dataset name"
)
parser.add_argument("--lr", type=float, default=1.41e-6, help="Dataset name")
parser.add_argument("--run_id", type=str, default=None, help="Run id to resume a run")
parser.add_argument(
    "--eval_script",
    type=str,
    default="scripts/sentiment_control/debug_evaluate.sh",
    help="Evaluation script to run after training",
)

args = parser.parse_args()
# Set seed
seed = 0
torch.manual_seed(seed)

# Setup reset signal
signal.signal(signal.SIGUSR1, _reset_handler)

# Initialize variables
train_config = TrainingConfig(
    num_epochs=args.num_epochs,
    policy_model=args.policy_model,
    base_models=args.base_models,
    dataset=args.dataset,
    learning_rate=args.lr,
    tracker_kwargs={
        "wandb": {"notes": args.description, "resume": "allow", "id": args.run_id}
    },
)
# Restore checkpoint if restarted
job_id = os.environ.get("SLURM_JOB_ID")
restart_count = int(os.getenv("SLURM_RESTART_COUNT", 0))
if job_id and restart_count > 0:
    print("Loading policy model from checkpoint")
    checkpoint_dir = os.path.join("checkpoints", job_id, "policy_models")
    policy_model_checkpoints = [
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
    ]
    policy_model_checkpoints.sort(key=os.path.getmtime, reverse=True)
    latest_checkpoint = policy_model_checkpoints[0]
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(latest_checkpoint)
else:
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        train_config.policy_model
    )
base_models = []
base_model_tokenizers = []
for base_model_name in train_config.base_models:
    base_models.append(AutoModelForCausalLM.from_pretrained(base_model_name))
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model_tokenizers.append(tokenizer)

train_dataset = load_from_disk(os.environ.get("DATASETS_PATH") + train_config.dataset)
if args.debug:
    debug_batch_size = 8
    train_dataset = train_dataset.select(range(debug_batch_size * 2))
    train_config.batch_size = debug_batch_size
    train_config.mini_batch_size = debug_batch_size
    train_config.project_name = "bbc-test"
reward_model = SentimentRewardModel()

ppo_trainer = prepare_ppo_trainer(policy_model, train_dataset, train_config)
ppo_trainer.accelerator.get_tracker("wandb").store_init_configuration(
    {"train_config": train_config}
)

if ppo_trainer.accelerator.is_main_process:
    print(train_config)
    run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
    start = time.time()
else:
    run_id = None

run_id = broadcast_object_list([run_id])[0]
print(f"Process ID: {ppo_trainer.accelerator.process_index}: Run ID : {run_id}")
train_config.run_id = run_id

logger = get_logger(__name__)
# Train policy model
ppo_trainer = train(
    ppo_trainer,
    base_models,
    base_model_tokenizers,
    [reward_model],
    logger,
    train_config,
)

# Save policy model
# if not args.debug and ppo_trainer.accelerator.is_main_process:
if args.debug and ppo_trainer.accelerator.is_main_process:
    # Create a directory for saved models if it doesn't exist
    run_id = ppo_trainer.accelerator.get_tracker("wandb").tracker._run_id
    if not job_id:
        save_dir = os.path.join(os.getcwd(), "checkpoints", run_id)
    else:
        save_dir = os.path.join(os.getcwd(), "checkpoints", job_id, "policy_models")

    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(
        save_dir, f"{train_config.policy_model}_{run_id}_{restart_count}"
    )
    ppo_trainer.save_pretrained(model_dir)
    print(f"Policy model saved at {model_dir}")

if ppo_trainer.accelerator.is_main_process:
    end = time.time()
    print(f"Training time: {timedelta(seconds=(end - start))}")
    start = end
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")

    if train_config.signal_reset:
        print("Requeuing job")
        os.system("scontrol requeue {}".format(os.environ.get("SLURM_JOB_ID")))
    else:
        print("Queuing evaluation script")
        os.system(f"sbatch {args.eval_script} {model_dir}")
