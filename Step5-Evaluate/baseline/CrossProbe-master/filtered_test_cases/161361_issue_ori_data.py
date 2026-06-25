import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
x = torch.randn(100, 1).to(device)
y = torch.randn(100, 1).to(device)
w = torch.rand(1, 1, requires_grad=True).to(device)
for epoch in range(10):
    pred = x @ w
    loss = torch.abs(y - pred).mean()
    loss.backward()
    with torch.no_grad():
        w.data -= 0.01 * w.grad
        w.grad.zero_()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")