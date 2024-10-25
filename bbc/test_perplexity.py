import pandas as pd
from train import perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

df = pd.read_csv("local_logs/s7qhpmoq/neutral_prompts_neg_log_s7qhpmoq.csv")

x = df.iloc[df.Perplexity.argmax()]

base_models = [AutoModelForCausalLM.from_pretrained("gpt2-large").cuda()]
tokenizers = [AutoTokenizer.from_pretrained("gpt2")]
tokenizers[0].pad_token = tokenizers[0].eos_token
prompts = [x.Prompt]
continuations = [[x.Continuation]]
print(perplexity(prompts, continuations, base_models, tokenizers))
