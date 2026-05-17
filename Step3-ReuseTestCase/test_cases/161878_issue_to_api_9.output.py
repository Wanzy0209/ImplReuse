import torch
import torch.nn as nn
import time
import unittest

def get_execution_step(step_tensor):
    """
    Helper function adapted from the logic of tf.compat.v1.train.global_step.
    
    Original TensorFlow logic:
      if context.executing_eagerly():
        return int(global_step_tensor.numpy())
        
    Translated PyTorch logic:
      Checks if the code is currently executing in a compiled (Inductor) context
      versus eager mode to handle the tensor appropriately.
    """
    # In PyTorch, we check if we are inside a torch.compile region
    if not torch.compiler.is_compiling():
        # Eager mode: equivalent to TF's executing_eagerly()
        return int(step_tensor.item())
    else:
        # Compiled mode (Inductor): return the tensor to be part of the graph
        return step_tensor

class SimpleBERTLayer(nn.Module):
    """
    A minimal representation of a BERT-like layer to reproduce the 
    performance regression scenario without requiring the full TorchBench setup.
    """
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=12, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        # Feed-forward network
        ff_out = self.linear1(x)
        ff_out = self.act(ff_out)
        ff_out = self.linear2(ff_out)
        
        return self.norm(x + ff_out)

class TestInductorPerformanceRegression(unittest.TestCase):
    def test_bert_amp_static_shape_performance(self):
        """
        Reproduces the performance regression scenario for BERT_pytorch on CPU
        using AMP and static shapes, while integrating the execution context 
        checking pattern from the similar API.
        """
        device = torch.device("cpu")
        model = SimpleBERTLayer().to(device)
        model.eval()
        
        # Static shape configuration (Batch size 2, Sequence length 128)
        # Matches the 'batch_size_new': 2 in the bug report
        batch_size = 2
        seq_len = 128
        hidden_dim = 768
        example_input = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        # Number of iterations for benchmarking
        num_iters = 50
        
        # --- 1. Eager Execution Benchmark ---
        # Warmup
        for _ in range(5):
            with torch.autocast("cpu"):
                _ = model(example_input)
                # Use the helper to mimic the global step retrieval logic
                _ = get_execution_step(torch.tensor(1))

        start_time = time.time()
        with torch.autocast("cpu"):
            for _ in range(num_iters):
                output = model(example_input)
                # Verify execution context logic in eager mode
                step = get_execution_step(torch.tensor(1))
                self.assertEqual(step, 1)
        eager_time = time.time() - start_time
        
        # --- 2. Inductor Execution Benchmark ---
        # Compile the model (Inductor backend)
        # The bug mentions 'cpp wrapper', which is often implied in deployment 
        # or specific benchmark harnesses, but torch.compile is the API trigger.
        compiled_model = torch.compile(model, mode="reduce-overhead")
        
        # Warmup for compiled model
        for _ in range(5):
            with torch.autocast("cpu"):
                _ = compiled_model(example_input)
                _ = get_execution_step(torch.tensor(1))

        start_time = time.time()
        with torch.autocast("cpu"):
            for _ in range(num_iters):
                output = compiled_model(example_input)
                # Verify execution context logic in compiled mode
                # Inside torch.compile, is_compiling() is True
                step = get_execution_step(torch.tensor(1))
        inductor_time = time.time() - start_time
        
        # Calculate Speedup
        # Bug report shows regression from ~2.59x to ~1.57x
        speedup = eager_time / inductor_time
        
        print(f"\nEager Time:     {eager_time:.4f}s")
        print(f"Inductor Time:  {inductor_time:.4f}s")
        print(f"Speedup:        {speedup:.2f}x")
        
        # Assertion to catch performance regressions.
        # We expect a speedup > 1.2. If it drops significantly (like to 1.0 or lower),
        # it indicates a severe regression similar to the issue.
        self.assertGreater(speedup, 1.2, 
                           f"Performance regression detected. Speedup {speedup:.2f}x is below threshold.")

if __name__ == "__main__":
    unittest.main()