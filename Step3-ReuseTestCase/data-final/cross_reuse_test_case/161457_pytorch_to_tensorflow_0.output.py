import torch
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import numpy as np

def test_meta_llama_tf_rewrite(self):
    """
    Adapted test case for tf.compat.v1.tpu.rewrite based on the 
    torch.compile meta-llama correctness issue.
    """
    # Enable bfloat16 policy to match the original bug's environment
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # Using a smaller model (distilgpt2) to ensure the test is runnable and minimal,
    # while preserving the logic of testing a causal language model.
    # Replace with "meta-llama/Llama-3.2-1B" if TF weights are available.
    model_name = "distilgpt2" 
    
    try:
        model = TFAutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        self.skipTest(f"Could not load model {model_name}: {e}")

    # Generate random inputs
    vocab_size = model.config.vocab_size
    input_ids = tf.random.uniform(
        (1, 1000), 
        minval=0, 
        maxval=vocab_size, 
        dtype=tf.int32
    )

    # Define the computation function to be rewritten
    def model_computation(input_ids):
        return model(input_ids).logits

    # 1. Eager Execution (Baseline)
    res1 = model_computation(input_ids)

    # 2. Compiled Execution using tf.compat.v1.tpu.rewrite
    # Note: This API compiles the computation for TPU.
    # We pass the computation and the list of input tensors.
    try:
        res2 = tf.compat.v1.tpu.rewrite(model_computation, [input_ids])
    except tf.errors.NotFoundError:
        # TPU not found in environment, skip TPU specific execution
        self.skipTest("TPU device not found. Skipping tf.compat.v1.tpu.rewrite test.")

    # 3. Verify Correctness
    # The original bug checks for closeness with atol=0.001
    # We convert tensors to numpy for comparison
    self.assertTrue(
        np.allclose(res1.numpy(), res2.numpy(), atol=0.001),
        "Eager and TPU rewrite results differ significantly"
    )