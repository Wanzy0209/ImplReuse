import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.profiler
import tempfile
import os

class FlexAttentionClosureModel(nn.Module):
    """
    Model that reproduces the graph break issue described in the bug report.
    The mask function captures a tensor from the forward pass scope.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(64, 64)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        processed = self.lin(x)
        
        # Intermediate tensor to be captured by the closure
        intermediate = processed.sum(dim=-1).detach()

        # This closure captures 'intermediate' and 'seq_len'.
        # This pattern is structurally similar to the closure defined
        # inside torch.profiler.tensorboard_trace_handler.
        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            threshold = intermediate[batch_idx, q_idx % seq_len]
            return (kv_idx <= q_idx) & (threshold > 0)

        block_mask = create_block_mask(
            mask_mod=mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=x.device,
            _compile=False,
        )

        q = processed.view(batch_size, 1, seq_len, self.dim)
        k = processed.view(batch_size, 1, seq_len, self.dim)
        v = processed.view(batch_size, 1, seq_len, self.dim)

        return flex_attention(q, k, v, block_mask=block_mask)

def test_flex_attention_with_profiler_handler():
    """
    Test case for Issue 164247.
    Verifies that the model runs under torch.compile (fullgraph) while
    being profiled using the similar API (tensorboard_trace_handler).
    """
    model = FlexAttentionClosureModel()
    
    # Compile with fullgraph=True to trigger the specific Dynamo graph break behavior
    compiled_model = torch.compile(model, fullgraph=True)
    
    input_tensor = torch.randn(2, 128, 64)

    # Leverage the similar API: torch.profiler.tensorboard_trace_handler
    # We use a temporary directory to store the trace output.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize the trace handler
        trace_handler = torch.profiler.tensorboard_trace_handler(tmpdir)
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=trace_handler
        ) as prof:
            # Run the compiled model
            output = compiled_model(input_tensor)

    # Assertions to verify the test ran successfully
    assert output is not None, "Model output is None"
    assert output.shape == (2, 1, 128, 64), f"Unexpected output shape: {output.shape}"
    
    # Verify that the profiler actually generated a trace file
    files = os.listdir(tmpdir)
    assert len(files) > 0, "Profiler did not generate trace files"

if __name__ == "__main__":
    test_flex_attention_with_profiler_handler()
    print("Test passed successfully.")