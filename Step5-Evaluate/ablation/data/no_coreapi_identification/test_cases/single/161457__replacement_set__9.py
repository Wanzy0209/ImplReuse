import torch
import tensorflow as tf
import copy
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_meta_llama_replacements():
    """
    Test case for torch.compile with meta-llama/Llama-3.2-1B.
    Replaces torch.randint with torch.bernoulli and torch.allclose with tf.compat.v1.image.resize_bicubic.
    """
    model_name = "meta-llama/Llama-3.2-1B"
    
    # Attempt to load the model; skip if not available or CUDA is not present
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda"
        )
    except Exception as e:
        print(f"Skipping test: Model or CUDA not available. Error: {e}")
        return

    model.eval()
    
    model.generation_config.do_sample = False
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    model.generation_config.max_new_tokens = 2000
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = 0.0

    vocab_size = tokenizer.vocab_size
    
    # Replacement 1: torch.randint -> torch.bernoulli
    # Original: torch.randint(low=0, high=vocab_size, size=(1, 1000), device="cuda", dtype=torch.long)
    # torch.bernoulli generates random 0s and 1s based on probability.
    # We create a probability tensor of 0.5 and cast the result to long.
    probs = torch.full((1, 1000), 0.5, device="cuda")
    input_ids = torch.bernoulli(probs).to(torch.long)
    
    example_inputs = {"input_ids": input_ids}

    model_copy = copy.deepcopy(model)

    start = time.time()
    res1 = model_copy(**example_inputs).logits
    print(f"Eager inference time: {time.time() - start}")

    model_copy = copy.deepcopy(model)
    model_copy.forward = torch.compile(model_copy.forward)

    start = time.time()
    res2 = model_copy(**example_inputs).logits
    print(f"Compiled inference time: {time.time() - start}")
    
    # Replacement 2: torch.allclose -> tf.compat.v1.image.resize_bicubic
    # Original: self.assertTrue(torch.allclose(res1, res2, atol=0.001))
    # tf.compat.v1.image.resize_bicubic expects a 4D tensor [batch, height, width, channels].
    # The logits are 3D [batch, seq_len, vocab_size]. We reshape them to 4D.
    # We apply the resize to the absolute difference to maintain a semantic link to comparison.
    
    diff = torch.abs(res1 - res2)
    # Convert to numpy and reshape for TensorFlow
    diff_np = diff.cpu().numpy().reshape(1, 1000, vocab_size, 1)
    
    # Perform the bicubic resize. Target size is arbitrary to satisfy API requirements.
    resized_diff = tf.compat.v1.image.resize_bicubic(diff_np, [500, vocab_size // 2])
    
    # Since the assertion logic is replaced by a transformation, we verify execution.
    print(f"Replacement API executed. Resized difference shape: {resized_diff.shape}")

if __name__ == "__main__":
    test_meta_llama_replacements()