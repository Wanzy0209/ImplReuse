import torch
import torch.nn as nn
import gc
import sys

# Check for flash_attn availability, as the bug is specific to this interaction
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    print("Skipping test: flash_attn is not installed.")
    sys.exit(0)

class FlashAttnModel(nn.Module):
    """
    A minimal model using flash_attn_varlen_func to reproduce the memory leak.
    """
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear layers to project inputs to Q, K, V
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cu_seqlens, max_seqlen):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Reshape for flash_attn_varlen_func: (total_seqlen, nheads, headdim)
        # We assume the input is already packed or we treat it as such for the varlen func
        q = q.transpose(1, 2).reshape(batch_size * seq_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2).reshape(batch_size * seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2).reshape(batch_size * seq_len, self.num_heads, self.head_dim)

        # Call the specific API mentioned in the bug report
        context = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
        )

        # Reshape back to (batch, seq_len, hidden_dim)
        context = context.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(context)

def get_tensor_count():
    """Helper to count live tensors, forcing garbage collection first."""
    gc.collect()
    return sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))

def test_torch_compile_memory_leak():
    """
    Test case for Issue 165407: Memory leak with torch.compile and flash_attn_varlen_func.
    """
    if not torch.cuda.is_available():
        print("Skipping test: CUDA is not available (flash_attn requires CUDA).")
        return

    device = "cuda"
    model = FlashAttnModel().to(device)
    
    # The API under test: torch.compile
    # Using default backend (inductor) which is implicated in the bug
    compiled_model = torch.compile(model)

    # Setup dummy inputs
    batch_size = 2
    seq_len = 32
    hidden_dim = 64
    
    # Create packed sequence inputs
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
    # Cumulative sequence lengths for varlen func
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)
    max_seqlen = seq_len

    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)

    print("Starting test loop to monitor tensor count...")
    initial_count = get_tensor_count()
    print(f"Initial Tensors: {initial_count}")

    # Run a few steps to observe the trend
    num_steps = 10
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass
        output = compiled_model(x, cu_seqlens, max_seqlen)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        optimizer.step()

        current_count = get_tensor_count()
        print(f"Step {step + 1} | Tensors: {current_count} | Delta: {current_count - initial_count}")

    final_count = get_tensor_count()
    total_increase = final_count - initial_count
    
    # Assertion: The number of tensors should not grow unboundedly.
    # We allow some buffer for internal caching, but a linear increase (as seen in the bug)
    # indicates a leak. The bug showed an increase of ~1000+ tensors over 150 steps.
    # Here we check if the increase is excessive for just 10 steps.
    assert total_increase < 200, (
        f"Potential memory leak detected. "
        f"Tensor count increased from {initial_count} to {final_count} "
        f"(delta: {total_increase})."
    )
    print("Test passed: No significant memory leak detected.")

if __name__ == "__main__":
    test_torch_compile_memory_leak()