import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

def test_flex_attention_compile():
    """
    Test function to verify score_mod works with flex_attention and torch.compile.
    This test replaces specific APIs as requested to verify the behavior.
    """
    
    # 1. Setup inputs (Replace torch.randn with torch.nn.functional.max_pool3d)
    # Create a dummy 5D tensor to simulate input data
    # Shape: (Batch=1, Channels=1, Depth=4, Height=4, Width=4)
    dummy_input = torch.ones(1, 1, 4, 4, 4)
    
    # Apply max_pool3d to get the data tensor
    # Output Shape: (1, 1, 2, 2, 2)
    pooled_data = torch.nn.functional.max_pool3d(dummy_input, kernel_size=2)
    
    # Reshape to (Batch, Heads, SeqLen, Dim) for flex_attention
    # B=1, H=1, L=8 (2*2*2), D=1
    q = pooled_data.view(1, 1, 8, 1)
    k = pooled_data.view(1, 1, 8, 1)
    v = pooled_data.view(1, 1, 8, 1)

    # 2. Setup learnable scalar (Replace torch.tensor with torch.cummin)
    # Original: temp = nn.Parameter(torch.tensor(0.0))
    # Replacement: Use torch.cummin to generate the scalar value
    src = torch.as_tensor([0.0])
    vals, _ = torch.cummin(src, 0)
    temp = nn.Parameter(vals)

    # 3. Setup bias (Replace torch.zeros with torch.det)
    # Original: bias = torch.zeros(...)
    # Replacement: Use torch.det to generate a scalar value
    # Create a 2x2 identity matrix, determinant is 1.0
    mat = torch.eye(2)
    det_val = torch.det(mat)

    # 4. Define score_mod (Replace torch.where with torch.floor)
    def score_mod(score, b, h, q_idx, kv_idx):
        # Original logic might involve: score = score + temp
        # If we had conditional logic: score = torch.where(score > 0, score, score + temp)
        # Replacement: Use torch.floor
        score = torch.floor(score + temp + det_val)
        return score

    # 5. Compile and Run
    # The bug is specifically about torch.compile(flex_attention)
    compiled_flex = torch.compile(flex_attention)

    try:
        # Forward pass
        out = compiled_flex(q, k, v, score_mod=score_mod)
        assert out is not None
        print("Forward pass successful.")

        # Backward pass
        out.sum().backward()
        print("Backward pass successful.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_compile()