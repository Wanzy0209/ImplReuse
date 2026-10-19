import unittest
import torch
import torch.nn as nn
import copy
import tensorflow as tf

# Enable TensorFlow eager execution for the metrics API
tf.compat.v1.enable_eager_execution()

class SimpleModel(nn.Module):
    """
    A minimal model to mimic the structure of a causal language model
    for testing torch.compile without external dependencies.
    """
    def __init__(self, vocab_size=32000):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        logits = self.fc(x)
        # Mimic the output structure of CausalLM (e.g., Llama)
        return type('Obj', (object,), {'logits': logits})()

class TestCompile(unittest.TestCase):
    def test_meta_llama_replacement(self):
        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vocab_size = 32000
        model = SimpleModel(vocab_size).to(device).to(torch.bfloat16)
        model.eval()

        # Replacement: torch.randint -> torch.full
        # Original: torch.randint(low=0, high=vocab_size, size=(1, 1000), ...)
        # New: torch.full((1, 1000), fill_value=1, ...)
        input_ids = torch.full((1, 1000), 1, device=device, dtype=torch.long)
        example_inputs = {"input_ids": input_ids}

        # Eager run
        model_copy = copy.deepcopy(model)
        res1 = model_copy(**example_inputs).logits

        # Compiled run
        model_copy = copy.deepcopy(model)
        model_copy.forward = torch.compile(model_copy.forward)
        res2 = model_copy(**example_inputs).logits

        # Replacement: torch.allclose -> tf.compat.v1.metrics.sparse_precision_at_k
        # We compare the precision of the results against the input labels.
        # For causal LM, we predict the next token, so we shift inputs.
        
        # Prepare data for TensorFlow
        # Labels: input_ids[:, 1:] (next tokens)
        # Predictions: logits[:, :-1, :] (predictions for current tokens)
        labels_torch = input_ids[:, 1:].cpu().numpy()
        preds1_torch = res1[:, :-1, :].cpu().numpy()
        preds2_torch = res2[:, :-1, :].cpu().numpy()

        labels_tf = tf.constant(labels_torch)
        preds1_tf = tf.constant(preds1_torch)
        preds2_tf = tf.constant(preds2_torch)

        # Calculate precision for eager result (res1)
        # Note: tf.compat.v1.metrics are stateful. In a real scenario, 
        # one might need to reset local variables between calls.
        precision1, update_op1 = tf.compat.v1.metrics.sparse_precision_at_k(
            labels=labels_tf, predictions=preds1_tf, k=1
        )
        # Run the update operation to get the metric value
        _ = update_op1.numpy()
        val1 = precision1.numpy()

        # Calculate precision for compiled result (res2)
        precision2, update_op2 = tf.compat.v1.metrics.sparse_precision_at_k(
            labels=labels_tf, predictions=preds2_tf, k=1
        )
        _ = update_op2.numpy()
        val2 = precision2.numpy()

        # Assert that the precision metrics are close, verifying consistency
        # between eager and compiled modes.
        self.assertAlmostEqual(val1, val2, places=3)

if __name__ == "__main__":
    unittest.main()