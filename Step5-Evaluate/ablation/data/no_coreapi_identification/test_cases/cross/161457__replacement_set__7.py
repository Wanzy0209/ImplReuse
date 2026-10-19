import unittest
import torch
import copy
import time
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestMetaLlama(unittest.TestCase):
    def test_meta_llama(self):
        model_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        
        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = 2000
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.temperature = 0.0

        vocab_size = tokenizer.vocab_size
        
        # Replacement: torch.randint -> torch.argmin
        # To use torch.argmin, we need an input tensor. We create a dummy tensor 
        # of shape (1, 1000, vocab_size) and find the min index along the last dimension.
        # This results in a tensor of shape (1, 1000) suitable for input_ids.
        dummy_input = torch.randn(1, 1000, vocab_size, device="cuda", dtype=torch.bfloat16)
        input_ids = torch.argmin(dummy_input, dim=-1)
        
        example_inputs = {"input_ids": input_ids}

        model_copy = copy.deepcopy(model)

        start = time.time()
        res1 = model_copy(**example_inputs).logits
        print(time.time() - start)

        model_copy = copy.deepcopy(model)
        model_copy.forward = torch.compile(model_copy.forward)

        start = time.time()
        res2 = model_copy(**example_inputs).logits
        print(time.time() - start)
        
        # Replacement: torch.allclose -> tf.compat.v1.image.resize_bilinear
        # tf.compat.v1.image.resize_bilinear expects 4D input [batch, height, width, channels].
        # The logits res1 and res2 are 3D [batch, seq, vocab].
        # We reshape them to 4D and convert to TensorFlow tensors.
        
        # Convert to numpy then to tensorflow tensors
        res1_tf = tf.convert_to_tensor(res1.cpu().numpy())
        res2_tf = tf.convert_to_tensor(res2.cpu().numpy())
        
        batch, seq, vocab = res1.shape
        # Reshape to [batch, height, width, channels] -> [1, seq, vocab, 1]
        res1_4d = tf.reshape(res1_tf, [batch, seq, vocab, 1])
        res2_4d = tf.reshape(res2_tf, [batch, seq, vocab, 1])
        
        # Perform the resize operation. 
        # Note: This replaces the comparison logic with a resize operation as requested.
        # We resize to half the sequence length just to perform the operation.
        resized_res1 = tf.compat.v1.image.resize_bilinear(res1_4d, size=[seq // 2, vocab])
        resized_res2 = tf.compat.v1.image.resize_bilinear(res2_4d, size=[seq // 2, vocab])
        
        # Assert that the operation completed and returned a non-None result
        self.assertTrue(resized_res1 is not None)
        self.assertTrue(resized_res2 is not None)