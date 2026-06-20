import torch
import unittest
import numpy as np

# Attempt to import TensorFlow and handle potential environment errors
# This catches the GLIBCXX version error which manifests as an ImportError
try:
    import tensorflow as tf
    from transformers import TFAutoModelForCausalLM, AutoTokenizer
    # Enable eager execution as per the similar API provided
    tf.compat.v1.enable_eager_execution()
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

@unittest.skipIf(not TF_AVAILABLE, "Skipping test: TensorFlow environment issue (ImportError/GLIBCXX)")
class TestMetaLlamaTF(unittest.TestCase):
    def test_meta_llama_tf(self):
        model_name = "meta-llama/Llama-3.2-1B"
        
        # Setup mixed precision to match the bfloat16 condition of the original bug
        # This ensures the model runs in bfloat16, similar to torch_dtype=torch.bfloat16
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForCausalLM.from_pretrained(
            model_name, 
            from_pt=True  # Load from PyTorch weights if necessary
        )
        
        vocab_size = tokenizer.vocab_size
        input_ids = tf.random.uniform(
            shape=(1, 1000),
            minval=0,
            maxval=vocab_size,
            dtype=tf.int32,
        )

        # 1. Run in Eager Mode (Baseline)
        # This corresponds to the uncompiled model in the original PyTorch test
        res1 = model(input_ids).logits

        # 2. Run in Graph Mode (Equivalent to torch.compile)
        # In TensorFlow, tf.function compiles a callable into a static graph
        compiled_model = tf.function(model)
        res2 = compiled_model(input_ids).logits

        # Compare results to check for correctness issues
        # The original test used atol=0.001
        self.assertTrue(np.allclose(res1.numpy(), res2.numpy(), atol=0.001))

if __name__ == '__main__':
    unittest.main()