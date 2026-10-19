# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
def test_meta_llama(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import time 
        import copy

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

        model_copy = copy.deepcopy(model)

        start = time.time()
        res1 = model_copy(**example_inputs).logits
        print(time.time() - start)

        model_copy = copy.deepcopy(model)
        model_copy.forward = torch.compile(model_copy.forward)

        start = time.time()
        res2 = model_copy(**example_inputs).logits
        print(time.time() - start)
        
        self.assertTrue(torch.allclose(res1, res2, atol=0.001))