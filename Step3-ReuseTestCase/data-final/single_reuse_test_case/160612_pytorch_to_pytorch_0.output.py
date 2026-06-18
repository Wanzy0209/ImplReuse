import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def test_prune_remove():
    # Create a linear module
    m = nn.Linear(5, 7)
    
    # Apply random unstructured pruning to the 'weight' parameter
    prune.random_unstructured(m, name="weight", amount=0.2)
    
    # Verify that pruning reparameterization attributes exist before removal
    assert hasattr(m, 'weight_orig'), "Original weight parameter 'weight_orig' should exist before removal"
    assert hasattr(m, 'weight_mask'), "Mask buffer 'weight_mask' should exist before removal"
    
    # Remove the pruning reparameterization
    # This permanently applies the pruning (mask is applied to the weight) 
    # and removes the 'weight_orig' and 'weight_mask' attributes.
    prune.remove(m, name="weight")
    
    # Verify that the reparameterization attributes are removed
    assert not hasattr(m, 'weight_orig'), "Original weight parameter 'weight_orig' should be removed"
    assert not hasattr(m, 'weight_mask'), "Mask buffer 'weight_mask' should be removed"
    
    # Verify that the 'weight' parameter still exists
    assert hasattr(m, 'weight'), "Weight parameter 'weight' should still exist"
    
    # Verify that the forward pre-hook is removed
    assert len(m._forward_pre_hooks) == 0, "Forward pre-hooks should be empty after removal"
    
    print("Test passed: prune.remove successfully removed reparameterization.")

if __name__ == "__main__":
    test_prune_remove()