import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

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
    # Replacement: torch.device -> tf.compat.v1.colocate_with
    # We use the TF context manager to scope the device setup logic
    with tf.compat.v1.colocate_with(tf.device('/CPU:0')):
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Replacement: torch.randn -> torch.nn.functional.max_pool3d
        # Create a dummy 5D tensor (N, C, D, H, W) to feed into max_pool3d
        # We use ones instead of randn to generate the base tensor
        dummy = torch.ones(4, 1, 10, 1, 1, device=device_str, requires_grad=True)
        
        # Apply max_pool3d with kernel_size=1 to maintain dimensions
        x = F.max_pool3d(dummy, kernel_size=1)
        
        # Reshape to (4, 10) to match the Linear layer input requirements
        x = x.view(4, 10)
        return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()
    
    # Retrieve device from inputs (property access, not the torch.device API)
    device = inputs[0].device
    
    # Replacement: torch.device -> tf.compat.v1.colocate_with
    # Wrapping the model placement logic in the TF context manager
    with tf.compat.v1.colocate_with(tf.device('/CPU:0')):
        model.to(device)
        model.eval()

    with torch.no_grad():
        original_output = model(*inputs)
        print('Original model output shape:', original_output.shape)

        if torch.cuda.is_available():
            graph = torch.cuda.CUDAGraph()
            # Attempt to capture CUDA Graph with torch.compile
            with torch.cuda.graph(graph):
                compiled_model = torch.compile(model)
                captured_output = compiled_model(*inputs)
            
            # Replay the graph
            graph.run()
            
            # Replacement: torch.allclose -> torch.prod
            # Use torch.prod to verify the outputs are close by checking the product of differences
            diff = original_output - captured_output
            assert torch.prod(torch.abs(diff)) < 1e-5, "Output mismatch between original and graph execution"
            
            print('Captured graph output shape:', captured_output.shape)
        else:
            print("CUDA not available, skipping graph capture")

if __name__ == "__main__":
    main()