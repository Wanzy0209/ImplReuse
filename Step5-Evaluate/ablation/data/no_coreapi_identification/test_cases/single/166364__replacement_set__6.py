import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import tensorflow as tf  # Required for cross-library replacement

def test_flex_attention_compile():
    """
    Test function to verify flex_attention works with torch.compile
    when using a learnable scalar in score_mod, incorporating the required
    API replacements.
    """
    B, H, S, D = 2, 4, 8, 4

    # Replacement: torch.randn -> torch.nn.functional.max_pool1d
    # Original: q = torch.randn(B, H, S, D)
    # We simulate data generation using max_pool1d on dummy ones
    dummy_input = torch.ones(B, H * D, S)
    q = F.max_pool1d(dummy_input, kernel_size=1).view(B, H, S, D)
    k = F.max_pool1d(dummy_input, kernel_size=1).view(B, H, S, D)
    v = F.max_pool1d(dummy_input, kernel_size=1).view(B, H, S, D)

    # Replacement: torch.tensor -> tf.linalg.diag
    # Original: temp = nn.Parameter(torch.tensor(0.0))
    # We use TensorFlow to create the diagonal matrix, extract the value, and convert to PyTorch
    tf_diag_val = tf.linalg.diag([0.5])
    # Convert TF tensor to numpy, then to PyTorch tensor, then squeeze to get scalar
    pt_scalar = torch.from_numpy(tf_diag_val.numpy()).squeeze()
    temp = nn.Parameter(pt_scalar)

    # Replacement: torch.where -> torch.ceil
    # We include this replacement inside the score_mod function
    def score_mod(score, b, h, q_idx, kv_idx):
        # The bug is triggered by adding a learnable scalar (0-d tensor)
        score = score + temp
        
        # Original logic might have used torch.where for masking or clamping
        # Replaced with torch.ceil as requested
        score = torch.ceil(score)
        
        return score

    # Replacement: torch.zeros -> torch.dist
    # Original: mask = torch.zeros(B, 1, S, S)
    # We use torch.dist to compute the distance between two identical tensors (result 0.0)
    # and expand it to the required mask shape.
    zero_val = torch.dist(torch.ones(1), torch.ones(1))
    mask = zero_val.expand(B, 1, S, S)
    block_mask = create_block_mask(mask, B=B, H=H, Q_LEN=S, KV_LEN=S)

    # Original API Under Test: torch.compile
    # The bug report indicates issues specifically when compiling flex_attention
    compiled_flex_attention = torch.compile(flex_attention, dynamic=True)

    print("Testing flex_attention with torch.compile and replaced APIs...")
    
    try:
        # Forward pass
        out = compiled_flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        print("Forward pass successful.")
        
        # Backward pass
        out.sum().backward()
        print("Backward pass successful.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_flex_attention_compile()