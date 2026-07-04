# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import numpy as np
import torch
import torchvision.models as models
from diffusers import DiffusionPipeline

from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel
from typing import Any, Callable, Dict, List, Optional, Tuple

seq_len = torch.export.Dim("seq_len", min=1, max=128)



def view_decomposition(x: torch.Tensor, size: List[torch.SymInt]) -> torch.Tensor:
    return torch.ops.aten._reshape_copy.default(x, size)

with torch.no_grad():
    config = Qwen3MoeConfig(num_hidden_layers=1, num_experts=4, use_cache=False, num_experts_per_tok=2)
    model = Qwen3MoeModel(config=config).half().cuda()
    inputs = {
        "input_ids": torch.randint(0, 128, (1, 12)).cuda(),
        "position_ids": torch.arange(12).unsqueeze(0).cuda(),
    }
    
    ep = torch.export.export(
                model,
                args=(inputs["input_ids"],),
                kwargs={"position_ids": inputs["position_ids"]},
                dynamic_shapes=({1: seq_len}, {1: seq_len}),
                strict=False,
                # allow_complex_guards_as_runtime_asserts=True,
            )
    decomp_table = torch.export.default_decompositions()
    decomp_table[torch.ops.aten.view.default] = view_decomposition

    after_decomp = ep.run_decompositions(decomp_table=decomp_table)
    print("Success")