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

def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    model = TestModel().cuda().eval()
    x = torch.randn(4, 10, device='cuda')
    
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled_model = torch.compile(model)
        output = compiled_model(x)
    
    graph.run()

if __name__ == "__main__":
    main()