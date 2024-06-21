import pudb
from transformers import AutoModelForCausalLM, AutoTokenizer

model_str = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_str, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_str)

strings = ["hi", "Hello, world!"]

pu.db
inputs = tokenizer(strings, padding=True, return_tensors="pt", padding_side="right")
target_ids = inputs.input_ids.clone()
target_ids[inputs.attention_mask == 0] = -100
outputs = model(**inputs, labels=inputs["input_ids"])
print(outputs.loss)

outputs = model.generate(**inputs)
prompt_continuation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
continuation = [y[len(x) :] for x, y in zip(strings, prompt_continuation)]

inputs = tokenizer(prompt_continuation, padding=True, return_tensors="pt")
target_ids = inputs.input_ids.clone()
attention_mask = inputs.attention_mask
target_ids[:, :4] = -100
outputs = model(
    input_ids=inputs["input_ids"], attention_mask=attention_mask, labels=target_ids
)

print(outputs.loss)

inputs = tokenizer(continuation, padding=True, return_tensors="pt")
target_ids = inputs.input_ids.clone()
attention_mask = inputs.attention_mask
target_ids[attention_mask == 0] = -100
outputs = model(
    input_ids=inputs["input_ids"], attention_mask=attention_mask, labels=target_ids
)
outputs = model(**inputs, labels=inputs["input_ids"])
print(outputs.loss)
