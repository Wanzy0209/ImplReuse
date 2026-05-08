import torch
import torch._inductor.config as config
config.cpp_wrapper = True
torch._logging.set_logs(output_code=True)

class SimpleMLP(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = torch.nn.Linear(28 * 28, 128)
    self.fc2 = torch.nn.Linear(128, 10)
    self.forward = torch.compile(self.forward)

  def forward(self, x):
    x = x.view(x.size(0), 28*28)
    x = torch.nn.functional.relu(self.fc1(x))
    x = self.fc2(x)
    return x

mlp = SimpleMLP().cuda()

with torch.profiler.profile(with_stack=True, activities=[
         torch.profiler.ProfilerActivity.CPU,
         torch.profiler.ProfilerActivity.CUDA,
    ], on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
) as prof:
  with torch.device("cuda"):
    for i in range(10):
      x = torch.randn(i, 28, 28).cuda()
      y = mlp(x)
      loss = y.sum()