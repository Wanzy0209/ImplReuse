import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def test_ln_structured_pruning():
    """
    Test case for torch.nn.utils.prune.ln_structured.
    Verifies that the function correctly prunes channels based on L-norm,
    creates the necessary mask and original parameter buffers, and modifies
    the module in place.
    """
    # 1. Setup: Create a simple Linear module
    # in_features=10, out_features=5
    module = nn.Linear(10, 5)
    
    # Store initial weight to verify 'weight_orig' later
    original_weight = module.weight.clone()

    # 2. Action: Apply ln_structured pruning
    # Prune 20% of the channels (rows in the weight matrix for Linear)
    # based on L2 norm (n=2) along dimension 0 (output channels)
    prune.ln_structured(module, name='weight', amount=0.2, n=2, dim=0)

    # 3. Verification: Check that the module was modified correctly
    
    # Check that the mask buffer was created
    assert hasattr(module, 'weight_mask'), "Mask 'weight_mask' not found in module"
    
    # Check that the original parameter was stored
    assert hasattr(module, 'weight_orig'), "Original parameter 'weight_orig' not found in module"
    assert torch.equal(module.weight_orig, original_weight), "Original weight does not match stored 'weight_orig'"

    # Check that the mask is binary (0s and 1s)
    mask = module.weight_mask
    assert torch.all((mask == 0) | (mask == 1)), "Mask contains non-binary values"

    # Check that the correct amount of channels were pruned
    # module.weight shape is (5, 10). dim=0 corresponds to the 5 output channels.
    # 20% of 5 is 1. So 1 channel (row) should be fully zeroed out.
    # 1 row * 10 columns = 10 zeros.
    num_zeros = (module.weight == 0).sum().item()
    expected_zeros = int(5 * 0.2) * 10
    assert num_zeros == expected_zeros, f"Expected {expected_zeros} zeros, found {num_zeros}"

    print("Test passed: torch.nn.utils.prune.ln_structured works as expected.")

if __name__ == "__main__":
    test_ln_structured_pruning()