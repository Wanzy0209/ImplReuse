import torch
optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(2, 2))], lr=torch.tensor(0.1))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
print('Before:', scheduler.get_last_lr())
optimizer.step()
scheduler.step()
print('After:', scheduler.get_last_lr())