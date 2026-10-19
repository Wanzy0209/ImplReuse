# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch                                 
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(
    "FacebookAI/xlm-roberta-base",           
    torch_dtype=torch.bfloat16,              
    device_map="auto",                       
    attn_implementation="sdpa"               
)                                            
print(model)