import torch
import torch.nn.functional as F

def main():
    # Generate random input probabilities and target values
    input_tensor = torch.rand(10, requires_grad=True)
    target_tensor = torch.rand(10)
    
    # Ensure input is between 0 and 1 for binary_cross_entropy
    input_tensor = torch.clamp(input_tensor, 0, 1)

    # Call the similar API: torch.nn.functional.binary_cross_entropy
    loss = F.binary_cross_entropy(input_tensor, target_tensor)

    # Verify the output is a scalar and non-negative
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss value should be non-negative"
    
    print(f"Test passed. Loss value: {loss.item()}")

if __name__ == "__main__":
    main()