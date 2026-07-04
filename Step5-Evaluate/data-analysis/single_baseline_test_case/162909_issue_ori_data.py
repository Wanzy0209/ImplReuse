# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Using device:", "xpu" if torch.xpu.is_available() else "cpu")

model_name = "./Qwen3-06B"  # path to local model

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # tried also device_map="sequential"
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)