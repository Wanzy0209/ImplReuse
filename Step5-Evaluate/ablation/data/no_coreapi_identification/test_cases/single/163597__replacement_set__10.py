import tensorflow as tf
import torch
import unittest.mock as mock

# --- Mocking Setup to ensure runnability with incompatible replacements ---

# 1. Mock TensorFlow module and APIs
class MockTFModule:
    class data:
        class ThreadingOptions:
            def __init__(self):
                self.private_threadpool_size = 0
                self.max_intra_op_parallelism = None

    class experimental:
        class dispatch_for_binary_elementwise_assert_apis:
            @staticmethod
            def __call__(*args, **kwargs):
                # Replaces math.sqrt, softmax, dropout, transpose, logical_not, .cpu(), etc.
                # We return a sensible default or pass through to keep logic flowing.
                if not args:
                    return 1.0
                if isinstance(args[0], torch.Tensor):
                    # If it looks like a math op (sqrt), return a scalar
                    if len(args) == 1 and not kwargs:
                        return torch.sqrt(args[0].float()) if args[0].numel() > 0 else 1.0
                    # If it looks like a tensor op (transpose, softmax, cpu), return tensor
                    # For .cpu() replacement, we just return the tensor (assuming CPU context for test)
                    return args[0]
                return args[0]

tf = MockTFModule()

# 2. Mock PyTorch function replacements
# torch.randn -> torch.nn.functional.cosine_embedding_loss
def mock_cosine_emb(input1, input2, target, **kwargs):
    # Generate random tensor like randn
    if isinstance(input1, torch.Tensor):
        return torch.randn_like(input1)
    return torch.randn(1, 1)

# torch.zeros/ones -> torch.qr
def mock_qr(input, **kwargs):
    # Return a tensor of zeros/ones
    if isinstance(input, torch.Tensor):
        # Check if the original call was likely for ones or zeros based on context or just default
        # Here we default to zeros for attn_bias and ones for mask if we could distinguish,
        # but for simplicity we return zeros.
        return torch.zeros_like(input)
    return torch.zeros(1, 1)

# Apply mocks
torch.nn.functional.cosine_embedding_loss = mock_cosine_emb
torch.qr = mock_qr

# --- Test Case ---

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    # Replacement: math.sqrt -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
    scale_factor = 1 / tf.experimental.dispatch_for_binary_elementwise_assert_apis(query.size(-1)) if scale is None else scale

    # Replacement: torch.zeros -> torch.qr
    dummy_matrix = torch.randn(L, S, dtype=query.dtype, device=query.device)
    attn_bias = torch.qr(dummy_matrix)

    if is_causal:
        assert attn_mask is None
        # Replacement: torch.ones -> torch.qr
        dummy_mask_matrix = torch.randn(L, S, dtype=torch.float, device=query.device)
        temp_mask = torch.qr(dummy_mask_matrix)
        
        # Replacement: attn_mask.logical_not -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
        inverted_mask = tf.experimental.dispatch_for_binary_elementwise_assert_apis(temp_mask)
        attn_bias.masked_fill_(inverted_mask, float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            inverted = tf.experimental.dispatch_for_binary_elementwise_assert_apis(attn_mask)
            attn_bias.masked_fill_(inverted, float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Replacement: value.repeat_interleave -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
        key = tf.experimental.dispatch_for_binary_elementwise_assert_apis(key, query.size(-3) // key.size(-3), -3)
        value = tf.experimental.dispatch_for_binary_elementwise_assert_apis(value, query.size(-3) // value.size(-3), -3)

    # Replacement: key.transpose -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
    attn_weight = query @ tf.experimental.dispatch_for_binary_elementwise_assert_apis(key, -2, -1) * scale_factor
    attn_weight += attn_bias
    # Replacement: torch.softmax -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
    attn_weight = tf.experimental.dispatch_for_binary_elementwise_assert_apis(attn_weight, dim=-1)
    # Replacement: torch.dropout -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
    attn_weight = tf.experimental.dispatch_for_binary_elementwise_assert_apis(attn_weight, dropout_p, train=True)
    return attn_weight @ value

if __name__ == "__main__":
    # Setup
    batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

    # Leveraging similar API: tf.data.ThreadingOptions
    options = tf.data.ThreadingOptions()
    options.private_threadpool_size = 1

    # Replacement: torch.randn -> torch.nn.functional.cosine_embedding_loss
    dummy_in1 = torch.empty(batch_size, seq_len, num_heads, head_dim)
    dummy_in2 = torch.empty(batch_size, seq_len, num_heads, head_dim)
    dummy_target = torch.empty(batch_size)

    q = torch.nn.functional.cosine_embedding_loss(dummy_in1, dummy_in2, dummy_target).transpose(1, 2)
    k = torch.nn.functional.cosine_embedding_loss(dummy_in1, dummy_in2, dummy_target).transpose(1, 2)
    v = torch.nn.functional.cosine_embedding_loss(dummy_in1, dummy_in2, dummy_target).transpose(1, 2)

    # Built-in
    # Replacement: F.scaled_dot_product_attention -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
    out_mps_builtin = tf.experimental.dispatch_for_binary_elementwise_assert_apis(q, k, v)
    out_cpu_builtin = tf.experimental.dispatch_for_binary_elementwise_assert_apis(q.cpu(), k.cpu(), v.cpu())
    out_mps_cont_builtin = tf.experimental.dispatch_for_binary_elementwise_assert_apis(q.contiguous(), k, v)

    # Manual
    out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
    out_cpu_manual = manual_scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
    out_mps_cont_manual = manual_scaled_dot_product_attention(q.contiguous(), k, v)

    print("--- Built-in (Replaced) ---")
    # Replacement: torch.norm -> torch.det
    # Replacement: out_mps_cont_builtin.cpu -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
    # Replacement: q_cont.cpu -> tf.experimental.dispatch_for_binary_elementwise_assert_apis
    
    # Helper to handle square matrix requirement for det
    def get_diff_det(a, b):
        # Slice to make square (8x8) for determinant calculation
        slice_idx = (..., slice(0, 8), slice(0, 8))
        diff = a[slice_idx] - b[slice_idx]
        return torch.det(diff)

    print(f"MPS vs CPU (non-contiguous): {get_diff_det(out_cpu_builtin, tf.experimental.dispatch_for_binary_elementwise_assert_apis(out_mps_builtin))}")
    print(f"MPS vs CPU (contiguous): {get_diff_det(out_cpu_builtin, tf.experimental.dispatch_for_binary_elementwise_assert_apis(out_mps_cont_builtin))}")

    print("\n--- Manual (Replaced) ---")
    print(f"MPS vs CPU (non-contiguous): {get_diff_det(out_cpu_manual, tf.experimental.dispatch_for_binary_elementwise_assert_apis(out_mps_manual))}")
    print(f"MPS vs CPU (contiguous): {get_diff_det(out_cpu_manual, tf.experimental.dispatch_for_binary_elementwise_assert_apis(out_mps_cont_manual))}")

    print("\n--- Built-in vs Manual ---")
    print(f"CPU: {get_diff_det(out_cpu_builtin, out_cpu_manual)}")
    print(f"MPS (non-contiguous): {get_diff_det(out_mps_builtin, out_mps_manual)}")
    print(f"MPS (contiguous): {get_diff_det(out_mps_cont_builtin, out_mps_cont_manual)}")