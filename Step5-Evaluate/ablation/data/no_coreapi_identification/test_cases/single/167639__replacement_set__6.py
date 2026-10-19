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
    # Replacement: torch.device -> tf.compat.v1.image.resize_bilinear
    # Original: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Note: This replacement changes the semantics significantly. 
    # We provide dummy arguments to make the function call valid.
    dummy_images = tf.constant([[[[1.0]]]])
    device = tf.compat.v1.image.resize_bilinear(dummy_images, size=[1, 1])

    # Replacement: torch.randn -> torch.nn.functional.max_pool1d
    # Original: x = torch.randn(4, 10, requires_grad=True, device=device)
    # Note: max_pool1d requires an input tensor. We create a dummy tensor to apply the pooling.
    # The resulting shape will differ from the original (4, 10).
    dummy_input = torch.randn(4, 10, 10, requires_grad=True)
    x = F.max_pool1d(dummy_input, kernel_size=2)
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

            # Replacement: torch.allclose -> torch.view_as_complex
            # Original: assert torch.allclose(original_output, graph_output)
            # Note: view_as_complex expects a tensor with a last dimension of size 2.
            # We stack the outputs to satisfy this requirement.
            stacked_output = torch.stack((original_output, graph_output), dim=-1)
            assert torch.view_as_complex(stacked_output)
        else:
            print("CUDA not available, skipping graph capture")

if __name__ == "__main__":
    main()