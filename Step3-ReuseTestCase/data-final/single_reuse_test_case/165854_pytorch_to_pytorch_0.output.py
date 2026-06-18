import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask

def test_compile_flex_attention_dynamic_captured_buffers():
    """
    Test case for Issue 165854: New flex flash path breaks w/ dynamic buffers.
    
    Verifies that torch.compile handles dynamic shapes correctly when the compiled
    function (flex_attention) uses a score_mod closure that captures a buffer
    whose size depends on the dynamic dimension (number of heads H).
    """
    device = "cuda"
    dtype = torch.float16
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    torch.manual_seed(0)

    # Compile flex_attention with fullgraph and dynamic=True
    # This is the API under test: torch.compile
    compiled_fa = torch.compile(flex_attention, fullgraph=True, dynamic=True)

    # Test with different head counts. 
    # Changing H forces dynamic shape handling.
    # The captured buffer (head_scale) must change size with H.
    head_counts = [4, 8, 4, 16, 4]
    
    B, S, D = 2, 256, 64

    for H in head_counts:
        # Create a captured buffer that depends on dynamic H
        head_scale = torch.randn(H, device=device, dtype=dtype, requires_grad=True)

        # Define score_mod capturing the dynamic buffer
        def score_mod(score, batch, head, token_q, token_kv):
            return score * head_scale[head]

        # Create inputs sized by H
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

        # Run forward pass
        outputs = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
        
        # Run backward pass
        loss = outputs.sum()
        loss.backward()

        # Basic sanity check to ensure execution completed without NaNs
        assert torch.isfinite(outputs).all(), f"Output contained NaNs/Infs for H={H}"
        assert torch.isfinite(loss).item(), f"Loss was NaN/Inf for H={H}"
        
        print(f" Successfully ran with H={H}")

if __name__ == "__main__":
    test_compile_flex_attention_dynamic_captured_buffers()