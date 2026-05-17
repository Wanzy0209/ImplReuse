import unittest
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestMetaLlamaWithTorchAll(unittest.TestCase):
    def test_meta_llama_correctness_with_torch_all(self):
        """
        Test case adapted to use torch.all for verification.
        Verifies the correctness of torch.compile on meta-llama/Llama-3.2-1B 
        by comparing eager and compiled results using torch.all.
        """
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
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(1, 1000),
            device="cuda",
            dtype=torch.long,
        )
        example_inputs = {"input_ids": input_ids}

        # Run eager model
        model_copy = copy.deepcopy(model)
        res1 = model_copy(**example_inputs).logits

        # Run compiled model
        model_copy = copy.deepcopy(model)
        model_copy.forward = torch.compile(model_copy.forward)
        res2 = model_copy(**example_inputs).logits
        
        # Adaptation: Replace torch.allclose with torch.all
        # Check if all elements in the difference tensor are within the absolute tolerance
        atol = 0.001
        diff = torch.abs(res1 - res2)
        self.assertTrue(torch.all(diff <= atol))

if __name__ == "__main__":
    unittest.main()