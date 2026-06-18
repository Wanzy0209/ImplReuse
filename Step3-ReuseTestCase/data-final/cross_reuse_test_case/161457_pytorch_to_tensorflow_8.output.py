import torch
import unittest
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

class TestMetaLlamaNameScope(unittest.TestCase):
    def test_meta_llama(self):
        """
        Adapted test case for tf.keras.name_scope based on the torch.compile bug report.
        The original issue was that torch.compile with bfloat16 caused accuracy errors.
        This test verifies that tf.keras.name_scope does not introduce similar accuracy issues
        when running the same model under similar conditions.
        """
        model_name = "meta-llama/Llama-3.2-1B"
        
        # Load tokenizer and model
        # Note: from_pt=True is used to load PyTorch weights into TensorFlow model
        # as TF weights might not be available for this specific model version.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForCausalLM.from_pretrained(
            model_name, 
            from_pt=True,
            # Replicating the bfloat16 condition from the bug report
            dtype=tf.bfloat16 
        )

        # Prepare inputs
        vocab_size = tokenizer.vocab_size
        # Using a smaller sequence length for minimal runnable example
        input_ids = tf.random.uniform(
            (1, 100), 
            minval=0, 
            maxval=vocab_size, 
            dtype=tf.int32
        )

        # Run 1: Baseline execution (Eager mode)
        res1 = model(input_ids).logits

        # Run 2: Execution inside tf.keras.name_scope
        # The original bug involved wrapping the model execution with torch.compile.
        # Here we wrap it with the similar API: tf.keras.name_scope.
        with tf.keras.name_scope("llama_inference_scope"):
            res2 = model(input_ids).logits

        # Verify correctness
        # Unlike the torch.compile bug where results differed, 
        # name_scope should not alter the numerical output.
        self.assertTrue(tf.reduce_all(tf.abs(res1 - res2) < 1e-3))

if __name__ == '__main__':
    unittest.main()