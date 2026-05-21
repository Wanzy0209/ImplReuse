import torch
import torch.nn as nn

def test_eval_crash_with_fake_quant():
    """
    Test case for Issue 167980: Jupyter kernel restart during eval after first epoch.
    
    This test preserves the original bug reproduction logic (Train -> Eval transition)
    while leveraging the similar API `torch.fake_quantize_per_channel_affine` 
    to investigate if the crash is related to quantization operations or tensor 
    manipulations handled by this API.
    """
    
    # Setup device (CPU is sufficient as the bug occurs on both CPU and CUDA)
    device = "cpu"
    
    # Define a model that integrates the similar API
    class QuantNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(784, 10)
            
            # Parameters for torch.fake_quantize_per_channel_affine
            # Scale and zero_point must be 1-D tensors for per-channel affine
            self.scale = torch.ones(784) 
            self.zero_point = torch.zeros(784, dtype=torch.int32)
            self.axis = 1 # Quantize along the feature dimension

        def forward(self, x):
            x = self.flatten(x)
            
            # Apply the similar API: fake_quantize_per_channel_affine
            # This operation is suspected to be related to the crash based on code similarity
            x = torch.fake_quantize_per_channel_affine(
                x, self.scale, self.zero_point, self.axis
            )
            
            return self.linear(x)

    # Initialize model, loss, and optimizer
    model = QuantNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create dummy data matching MNIST dimensions (Batch=80, 1, 28, 28)
    # Using random data instead of downloading MNIST to keep the test minimal and runnable
    X_train = torch.randn(80, 1, 28, 28).to(device)
    y_train = torch.randint(0, 10, (80,)).to(device)
    
    X_test = torch.randn(80, 1, 28, 28).to(device)
    y_test = torch.randint(0, 10, (80,)).to(device)

    print("=== TRAIN ===")
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("=== EVAL (CRASH CHECK) ===")
    model.eval()
    with torch.no_grad():
        # The original bug report indicates the kernel restarts here during evaluation
        # after the first training epoch.
        pred = model(X_test)
        
        # If we reach this point, the kernel did not crash for this specific configuration
        assert pred.shape == (80, 10), "Output shape mismatch"
        print("Eval step completed successfully.")

if __name__ == "__main__":
    test_eval_crash_with_fake_quant()