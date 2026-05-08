import torch

class Isin(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(777)
        self.device = device
        self.x = torch.randint(-50, 50, (1,), dtype=torch.int64, device=device)
        self.y = torch.randint(-50, 50, (), dtype=torch.int64, device=device)

    def forward(self):
        print(self.x)
        print(self.y)
        out = torch.isin(self.x, self.y, assume_unique=False, invert=False)
        return {'out': out}

model = Isin('cuda')
print("Eager:", model.forward())
compiled_model = torch.compile(model, backend='inductor')
print("Inductor:", compiled_model.forward())