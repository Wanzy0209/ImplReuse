import torch
import torch.nn as nn

# Keeping the structure similar to the original test case for consistency
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_default_model():
    return TestModel()

def get_sample_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(4, 10, requires_grad=True, device=device)
    return (x,)

def main():
    # We use the inputs to test torch.any
    inputs = get_sample_inputs()
    device = inputs[0].device
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    with torch.no_grad():
        # Original API: torch.compile
        # Similar API: torch.any
        # We test torch.any on the input tensor directly
        
        # Eager execution
        original_output = torch.any(inputs[0])
        print('Original output:', original_output.item())

        # CUDA Graph capture
        graph = torch.cuda.CUDAGraph()
        # Placeholder for the output captured in the graph
        captured_output = None

        with torch.cuda.graph(graph):
            # Replace the torch.compile call with torch.any
            captured_output = torch.any(inputs[0])

        # Replay the graph
        graph.run()

        print('Captured graph output:', captured_output.item())

        # Verify
        assert torch.equal(original_output, captured_output), "Output mismatch between original and graph execution"
        print("Test passed: torch.any is compatible with CUDA Graph capture")

if __name__ == "__main__":
    main()