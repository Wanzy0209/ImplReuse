import torch
import torch.nn.functional as F

def test_binary_cross_entropy():
    """
    Test case for torch.nn.functional.binary_cross_entropy.
    Adapted from the context of the FSDP2 example to verify the loss calculation.
    """
    # Setup device (using CPU to ensure runnability without specific GPU requirements)
    device = torch.device("cpu")
    
    # Define dimensions similar to the original example
    batch_size = 4
    vocab_size = 1024
    
    # Create dummy input (logits) and target tensors
    # Note: binary_cross_entropy expects probabilities, so we apply sigmoid
    input_logits = torch.randn(batch_size, vocab_size, device=device)
    input_probs = torch.sigmoid(input_logits)
    target = torch.rand(batch_size, vocab_size, device=device)
    
    # Optional: weight tensor
    weight = torch.ones(vocab_size, device=device)

    # Test 1: Default reduction ('mean')
    loss_mean = F.binary_cross_entropy(input_probs, target)
    assert isinstance(loss_mean, torch.Tensor), "Output should be a Tensor"
    assert loss_mean.dim() == 0, "Default reduction should result in a scalar"
    assert not torch.isnan(loss_mean), "Loss should not be NaN"
    print(f"Test 1 Passed: Mean reduction loss = {loss_mean.item()}")

    # Test 2: Reduction 'sum'
    loss_sum = F.binary_cross_entropy(input_probs, target, reduction='sum')
    assert loss_sum.dim() == 0, "Sum reduction should result in a scalar"
    assert loss_sum.item() >= 0, "Loss sum should be non-negative"
    print(f"Test 2 Passed: Sum reduction loss = {loss_sum.item()}")

    # Test 3: Reduction 'none'
    loss_none = F.binary_cross_entropy(input_probs, target, reduction='none')
    assert loss_none.shape == (batch_size, vocab_size), "No reduction should match input shape"
    print(f"Test 3 Passed: None reduction shape = {loss_none.shape}")

    # Test 4: With weights
    loss_weighted = F.binary_cross_entropy(input_probs, target, weight=weight)
    assert loss_weighted.dim() == 0, "Weighted loss should be a scalar"
    print(f"Test 4 Passed: Weighted loss = {loss_weighted.item()}")

    print("All tests for binary_cross_entropy passed successfully.")

if __name__ == "__main__":
    test_binary_cross_entropy()