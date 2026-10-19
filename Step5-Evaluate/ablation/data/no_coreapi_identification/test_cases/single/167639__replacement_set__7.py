import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import tensorflow as tf
import os

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
    # Replacement: torch.device -> torch.distributed.recv
    # Original: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize distributed if not already done (required for recv)
    if not dist.is_initialized():
        try:
            # Using a simple TCP init for standalone execution attempt
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)
        except Exception:
            pass # Ignore if environment is not set up for distributed

    # Perform the replacement API call
    # recv requires a tensor to receive into. We create a dummy one.
    dummy_tensor = torch.zeros(1)
    recv_result = torch.distributed.recv(dummy_tensor)
    
    # To keep the test case runnable and maintain logic (getting a device object),
    # we check the result. If recv didn't return a device object (which it won't),
    # we fall back to the standard logic to ensure the script continues.
    if isinstance(recv_result, torch.device):
        device = recv_result
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Replacement: torch.randn -> torch.nn.functional.max_pool2d
    # Original: x = torch.randn(4, 10, requires_grad=True, device=device)
    
    # max_pool2d requires 4D input (N, C, H, W). We create a dummy input.
    # We use kernel_size=1 to preserve dimensions roughly similar to the original intent.
    dummy_input = torch.ones(4, 1, 10, 1, device=device, requires_grad=True)
    
    # Perform the replacement API call
    x = torch.nn.functional.max_pool2d(dummy_input, kernel_size=1)
    
    # Reshape back to (4, 10) to match the model's expected input shape
    x = x.squeeze(1).squeeze(1)
    
    return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()
    device = inputs[0].device
    model.to(device)
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
            
            # Replacement: torch.allclose -> tf.compat.v1.image.resize_bilinear
            # Original: assert torch.allclose(original_output, graph_output)
            
            # resize_bilinear expects 4D tensor [batch, height, width, channels]
            # We reshape the outputs to fit the API signature.
            orig_4d = original_output.unsqueeze(-1).unsqueeze(-1).float()
            graph_4d = graph_output.unsqueeze(-1).unsqueeze(-1).float()
            
            # Perform the replacement API call
            # Note: This performs a resize operation instead of a comparison.
            _ = tf.compat.v1.image.resize_bilinear(orig_4d, size=(2, 2))
            _ = tf.compat.v1.image.resize_bilinear(graph_4d, size=(2, 2))
            
            print("Replaced torch.allclose with tf.compat.v1.image.resize_bilinear")
        else:
            print("CUDA not available, skipping graph capture")

if __name__ == "__main__":
    main()