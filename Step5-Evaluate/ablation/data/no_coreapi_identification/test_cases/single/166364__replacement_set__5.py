import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

def test_flex_attention_with_replacements(use_compile=False):
    """
    Test function to verify flex_attention works with specific API replacements.
    Replaces:
    - torch.randn -> torch.nn.functional.adaptive_max_pool3d
    - torch.tensor -> torch.diag (semantics of tf.keras.ops.diag)
    - torch.where -> torch.scatter
    - torch.zeros -> torch.norm
    """
    B, H, M, K = 2, 2, 4, 4

    # 1. Replace torch.randn with torch.nn.functional.adaptive_max_pool3d
    # Create a dummy 5D tensor to pool from
    # Shape (N, C, D, H, W) -> (B*H, 1, M*2, K*2, 1)
    dummy_input = torch.ones(B * H, 1, M * 2, K * 2, 1)
    # Pool to (B*H, 1, M, K, 1)
    pooled = F.adaptive_max_pool3d(dummy_input, (M, K, 1))
    # Reshape to (B, H, M, K) for flex_attention
    q = pooled.squeeze(1).squeeze(-1).view(B, H, M, K)
    k = pooled.squeeze(1).squeeze(-1).view(B, H, M, K)
    v = pooled.squeeze(1).squeeze(-1).view(B, H, M, K)

    # 2. Replace torch.tensor with torch.diag (semantics of tf.keras.ops.diag)
    # Original: temp = nn.Parameter(torch.tensor(0.0))
    # New: Create a 1x1 diagonal matrix parameter.
    # Note: This changes the scalar to a 1x1 tensor.
    temp = nn.Parameter(torch.diag(torch.ones(1)))

    # 3. Replace torch.where with torch.scatter
    # Simulating a selection/update operation using scatter
    # Create an index to scatter values into q
    index = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    src = torch.ones(1, 1, 1, 1)
    # Scatter src into q at index
    q = q.scatter(0, index, src)

    def score_mod(score, b, h, q, kv):
        # score is (B, H, M, K). temp is (1, 1).
        # Broadcasting should work if supported.
        score = score + temp
        return score

    # Apply torch.compile if requested
    if use_compile:
        flex_attention = torch.compile(flex_attention)

    # Run forward
    out = flex_attention(q, k, v, score_mod=score_mod)

    # 4. Replace torch.zeros with torch.norm
    # Original code might use zeros for initialization or loss.
    # Here we use norm to calculate loss.
    loss = torch.norm(out)
    
    # Run backward
    loss.backward()

    # Assertions
    assert temp.grad is not None, "Learnable parameter should have gradients"
    print(f"Test passed with use_compile={use_compile}")

if __name__ == "__main__":
    print("Testing without compile...")
    test_flex_attention_with_replacements(use_compile=False)
    print("Testing with compile...")
    test_flex_attention_with_replacements(use_compile=True)