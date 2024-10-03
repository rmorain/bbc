import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

df = pd.read_csv("local_logs/4z4wflco/negative_prompts_pos_log_4z4wflco.csv")
# base_model = AutoModelForCausalLM.from_pretrained("gpt2-large")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
hist = df.hist(bins=5)
