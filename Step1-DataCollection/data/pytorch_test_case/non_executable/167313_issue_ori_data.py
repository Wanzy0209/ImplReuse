In [1]: import torch

In [2]: x = torch.rand(2, device="cuda")
         
In [3]: a = torch.rand(2, 3, device="cuda")

In [4]: b = torch.rand(3, 2, device="cuda")

In [5]: f = lambda x, a, b: torch.nn.functional.relu(torch.addmm(x, a, b, alpha=0.5, beta=0.5))

In [6]: fc = torch.compile(f)

In [7]: f(x, a, b)
Out[7]: 
tensor([[0.7411, 0.6704],
        [0.5025, 0.4090]], device='cuda:0')

In [8]: fc(x, a, b)
Out[8]: 
tensor([[1.4822, 1.3408],
        [1.0049, 0.8180]], device='cuda:0')