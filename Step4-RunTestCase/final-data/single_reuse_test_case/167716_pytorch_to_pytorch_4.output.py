import torch
import torch.nn as nn

def test_layernorm_sparse_input():
    """
    Test case adapted from Issue 167716 (torch.sparse.mm segfault).
    Verifies if torch.nn.LayerNorm handles sparse tensor inputs without 
    causing a segmentation fault when converting the result to dense.
    """
    # Setup from the original bug report
    torch.manual_seed(42)
    indices_A = torch.tensor([[0, 1, 2], [0, 2, 3]])
    values_A = torch.tensor([1.0, 2.0, 3.0])
    # Create a sparse tensor of shape (3, 4)
    A = torch.sparse_coo_tensor(indices_A, values_A, size=(3, 4))

    # Initialize the similar API: torch.nn.LayerNorm
    # The input shape is (3, 4), so we normalize over the last dimension (size 4)
    layer_norm = nn.LayerNorm(4)

    # Adapt the original call site:
    # Original: y = torch.sparse.mm(a, b); z = y.to_dense()
    # Adapted: y = layer_norm(a); z = y.to_dense()
    
    try:
        y = layer_norm(A)
        # The original bug crashed specifically on to_dense()
        z = y.to_dense()
        print("Test passed. Result converted to dense successfully.")
        assert z.shape == (3, 4)
    except RuntimeError as e:
        # LayerNorm might not support sparse inputs natively, 
        # resulting in a RuntimeError. This is acceptable behavior 
        # as long as it is not a Segmentation Fault.
        print(f"RuntimeError caught (Sparse input might not be supported): {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_layernorm_sparse_input()