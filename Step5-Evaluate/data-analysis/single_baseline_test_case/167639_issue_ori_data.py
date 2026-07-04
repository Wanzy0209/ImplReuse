# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn as nn

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
            assert torch.allclose(original_output, graph_output), "Output mismatch between original and graph execution"
        else:
            print("CUDA not available, skipping graph capture")

if __name__ == "__main__":
    main()