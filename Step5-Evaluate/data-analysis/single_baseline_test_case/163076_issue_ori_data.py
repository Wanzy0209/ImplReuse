# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
from torch.nn import Module
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_NAME = "Qwen/Qwen3-0.6B"
ONNX_PATH = "qwen3_kv_cache.onnx"
OPSET = 18  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).eval().cuda()

device = model.device

input_ids = tokenizer.encode("Hello World", return_tensors="pt").to(device)

class QwenTorchModule(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, past_key_values=None, **kwargs):
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            **kwargs
        )
        return (outputs.logits, outputs.past_key_values)

qwen_module = QwenTorchModule(model)

with torch.inference_mode():
    outputs = qwen_module(input_ids)
    past_key_values = outputs[1]

old_forward = qwen_module.forward
def _export_forward(input_ids, past_key_values):
    return old_forward(input_ids=input_ids, past_key_values=past_key_values)

qwen_module.forward = _export_forward

export_inputs = (input_ids, past_key_values)

torch.onnx.export(
    qwen_module,
    export_inputs,
    ONNX_PATH,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=["input_ids", "past_key_values"],
    output_names=["logits", "present_key_values"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
)