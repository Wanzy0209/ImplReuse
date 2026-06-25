import torch
import os
from transformers import Llama4Config, Llama4ForConditionalGeneration, AutoTokenizer

DEVICE = "cuda"
MODEL_CONFIG_NAME = "4-Func.json"
PROMPT = "The future of AI is"
INPUT_ID_LENGTH = 128
MAX_NEW_TOKENS = 10

if not os.path.exists(MODEL_CONFIG_NAME):
    print(f"Config file {MODEL_CONFIG_NAME} not found")
    exit()

model_config = Llama4Config.from_json_file(MODEL_CONFIG_NAME)
model = Llama4ForConditionalGeneration._from_config(model_config).to(DEVICE).eval()
model.forward = torch.compile(model.forward, dynamic=True, fullgraph=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
inputs = tokenizer(PROMPT, return_tensors="pt", max_length=INPUT_ID_LENGTH, truncation=True, padding="max_length").to(DEVICE)
batch_inputs = {k: v.expand(4, -1) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**batch_inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True, do_sample=False)