import torch
import torch.nn as nn

class SimpleModel(nn.Module):
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

def test_torch_compile_cuda_graph():
    """
    Test case to verify torch.compile compatibility with CUDA Graph capture.
    This reproduces the issue where torch.compile attempts to access CUDA RNG state
    during compilation inside the graph capture context.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    device = torch.device('cuda')
    model = SimpleModel().to(device)
    model.eval()

    # Prepare sample inputs
    inputs = torch.randn(4, 10, device=device)

    # Run original model to get baseline output
    with torch.no_grad():
        original_output = model(inputs)
        print('Original model output shape:', original_output.shape)

    # Attempt to capture CUDA Graph with torch.compile
    # Bug ID: 167639 - torch.compile incompatible with CUDA Graph capture
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # The compilation happens here. If the bug exists, this will fail
            # because the backend (e.g., inductor) tries to access RNG state.
            compiled_model = torch.compile(model)
            captured_output = compiled_model(inputs)
        
        # Replay the graph
        graph.replay()
        print('Captured graph replay successful.')
        
        # Verify output shape consistency
        assert captured_output.shape == original_output.shape, "Output shape mismatch"
        print("Test passed: torch.compile is compatible with CUDA Graph capture.")

    except RuntimeError as e:
        print(f"Test failed with RuntimeError: {e}")
        print("This indicates the bug where torch.compile accesses RNG state during graph capture.")

if __name__ == "__main__":
    test_torch_compile_cuda_graph()