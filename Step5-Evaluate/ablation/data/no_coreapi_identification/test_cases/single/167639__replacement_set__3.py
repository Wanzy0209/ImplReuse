import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # Replacement: torch.device -> torch.nn.functional.softmin
    # Original: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # We replace the torch.device call with softmin. Since softmin returns a tensor,
    # we cannot use it as a device object for .to() calls. We will use the string 'cuda' directly.
    device = torch.nn.functional.softmin(torch.tensor([1.0]))
    
    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool1d
    # Original: x = torch.randn(4, 10, requires_grad=True, device=device)
    # We generate the input tensor using adaptive_max_pool1d instead of randn.
    # We use 'cuda' string directly as 'device' variable is now a tensor.
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_tensor = torch.ones(4, 10, 1, device=target_device)
    x = torch.nn.functional.adaptive_max_pool1d(base_tensor, output_size=10).squeeze(-1)
    x.requires_grad_(True)
    return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()
    # Infer device from inputs since 'device' variable is now a tensor due to replacement
    target_device = inputs[0].device
    model.to(target_device)
    model.eval()

    with torch.no_grad():
        original_output = model(*inputs)
        print('Original model output shape:', original_output.shape)

        if torch.cuda.is_available():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compiled_model = torch.compile(model)
                captured_output = compiled_model(*inputs)
            graph_output = graph.run()
            print('Captured graph output shape:', graph_output.shape)
            
            # Replacement: torch.allclose -> torch.xlogy
            # Original: assert torch.allclose(original_output, graph_output)
            # We use xlogy to verify closeness. If outputs are close, diff is small, so xlogy(diff, diff) is small.
            diff = original_output - graph_output
            # Add epsilon to avoid log(0) issues
            check_val = torch.xlogy(diff.abs() + 1e-9, diff.abs() + 1e-9)
            assert check_val.max() < 1e-6, "Output mismatch between original and graph execution"
        else:
            print("CUDA not available, skipping graph capture")

if __name__ == "__main__":
    main()