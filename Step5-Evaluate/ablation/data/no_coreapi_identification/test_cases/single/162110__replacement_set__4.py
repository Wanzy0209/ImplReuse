import numpy as np
import torch
import tensorflow as tf
import torchvision.models as models
from diffusers import DiffusionPipeline

from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel
from typing import Any, Callable, Dict, List, Optional, Tuple

# torch.export.Dim -> tf.compat.v1.train.global_step
seq_len = tf.compat.v1.train.global_step(None, None)

# torch.Tensor -> tf.compat.v1.train.global_step
# torch.SymInt -> tf.compat.v1.train.global_step
def view_decomposition(x: tf.compat.v1.train.global_step, size: List[tf.compat.v1.train.global_step]) -> tf.compat.v1.train.global_step:
    # torch.ops.aten._reshape_copy.default -> tf.compat.v1.train.global_step
    return tf.compat.v1.train.global_step(x, size)

# torch.no_grad -> tf.compat.v1.train.global_step
with tf.compat.v1.train.global_step():
    config = Qwen3MoeConfig(num_hidden_layers=1, num_experts=4, use_cache=False, num_experts_per_tok=2)
    model = Qwen3MoeModel(config=config).half().cuda()
    inputs = {
        # torch.randint -> torch.fmax
        # torch.randint(0, 128, (1, 12)) -> torch.fmax(torch.tensor(0), torch.tensor(128))
        "input_ids": torch.fmax(torch.tensor(0), torch.tensor(128)).cuda(),
        # torch.arange -> torch.quantile
        # torch.arange(12) -> torch.quantile(torch.tensor([12]), torch.tensor(0.5))
        "position_ids": torch.quantile(torch.tensor([12]), torch.tensor(0.5)).unsqueeze(0).cuda(),
    }
    
    # torch.export.export -> tf.debugging.enable_check_numerics
    ep = tf.debugging.enable_check_numerics()

    # torch.export.default_decompositions -> tf.compat.v1.train.global_step
    decomp_table = tf.compat.v1.train.global_step(None, None)

    # torch.ops.aten.view.default -> tf.compat.v1.train.global_step
    decomp_table[tf.compat.v1.train.global_step] = view_decomposition

    after_decomp = ep.run_decompositions(decomp_table=decomp_table)
    print("Success")