import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

def test_pipeline_vs_single():
    # Single GPU baseline
    model = SimpleModel()
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)
    
    # Forward
    out = model(x)
    loss = nn.MSELoss()(out, y)
    
    # Backward
    loss.backward()
    single_grad_norm = model.linear.weight.grad.norm().item()
    
    # Pipeline version (simplified)
    # This would need proper pipeline setup with ScheduleGPipe
    # Compare gradient norms
    
    print(f"Single GPU grad norm: {single_grad_norm}")
    # Pipeline grad norm should be similar

if __name__ == "__main__":
    test_pipeline_vs_single()